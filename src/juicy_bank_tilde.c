/*
 * Bela + Pure Data OLED UI wrapper for SSD1306
 * Based on the official libpd custom-render example.
 *
 * IMPORTANT:
 * This version receives screen state through Bela/libpd bound float receivers.
 * The synth sends directly to the named receivers internally, so no Pd UI outlet/router is needed.
 *
 *   [s bela_screen_page]        <- float page index
 *   [s bela_screen_selected]    <- float selected parameter 0..5
 *   [s bela_screen_preset_slot] <- float preset slot number
 *   [s bela_screen_param0]      <- float value
 *   [s bela_screen_param1]      <- float value
 *   [s bela_screen_param2]      <- float value
 *   [s bela_screen_param3]      <- float value
 *   [s bela_screen_param4]      <- float value
 *   [s bela_screen_param5]      <- float value
 *
 * It does NOT rely on libpd message hooks, only floatHook + bindSymbols.
 *
 * PATCHED: preset_slot no longer overwrites param0 storage.
 */

#include <Bela.h>
#include <libraries/BelaLibpd/BelaLibpd.h>

#include <string.h>
#include <cmath>
#include <string>
#include <atomic>
#include <mutex>
#include <algorithm>
#include <chrono>
#include <array>

#include <fcntl.h>
#include <unistd.h>
#include <sys/ioctl.h>
#include <linux/i2c-dev.h>

// ------------------------------------------------------------
// OLED constants
// ------------------------------------------------------------
static constexpr int kOledWidth = 128;
static constexpr int kOledHeight = 64;
static constexpr int kI2CBus = 1;
static constexpr int kI2CAddress = 0x3C;
static constexpr int kParamCount = 6;

// ------------------------------------------------------------
// Page enum (must match your synth workflow page numbers)
// ------------------------------------------------------------
enum ScreenPage {
	kPagePlay = 0,
	kPagePlayAlt,
	kPageBodyA1,
	kPageBodyA2,
	kPageBodyB1,
	kPageBodyB2,
	kPageDampers,
	kPageExciterA,
	kPageExciterB,
	kPageSpace,
	kPageEcho,
	kPageSaturation,
	kPageModLfo1,
	kPageModLfo2,
	kPageVelocity,
	kPagePressure,
	kPageGlobalEdit,
	kPageResonatorEdit,
	kPagePreset,
	kPageCount
};

struct ScreenState {
	int page = 0;
	int selected = 0;
	int presetSlot = 0;
	int presetMode = 0;
	int presetCursor = 0;
	int presetUsed = 0;
	int patchDirty = 0;
	int feedback = 0;
	char presetName[17] = "                ";
	float values[kParamCount] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
	std::atomic<bool> dirty{true};
	std::mutex mutex;
};

static ScreenState gScreenState;
static AuxiliaryTask gOledTask;
static std::atomic<bool> gOledReady{false};

static inline uint64_t uiNowMs()
{
	using namespace std::chrono;
	return (uint64_t)duration_cast<milliseconds>(steady_clock::now().time_since_epoch()).count();
}

static std::array<uint64_t, kParamCount> gParamChangedMs = {0, 0, 0, 0, 0, 0};
static uint64_t gIgnoreParamUntilMs = 0;
static int gOverlayTouchNonce = -1;
static int gOverlayPage = -1;
static int gOverlayParam = -1;
static uint64_t gOverlayUntilMs = 0;

static float gExciterACache[kParamCount] = {0.f, 0.12f, 0.22f, 0.65f, 0.24f, 0.5f};
static float gExciterBCache[kParamCount] = {0.35f, 0.f, 0.f, 0.f, 0.f, 0.f};
static float gSpaceSmooth[kParamCount] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static float gEchoSmooth[kParamCount] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static float gSatSmooth[kParamCount] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
static float gLfoSmooth[2][kParamCount] = {{0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}};
static float gPerfSmooth[2][kParamCount] = {{0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}};
static float gMacroSmooth[3][kParamCount] = {{0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}, {0.f, 0.f, 0.f, 0.f, 0.f, 0.f}};

// ------------------------------------------------------------
// Tiny SSD1306 driver
// ------------------------------------------------------------
class SSD1306 {
public:
	bool setup(int bus, int address)
	{
		char path[32];
		snprintf(path, sizeof(path), "/dev/i2c-%d", bus);
		fd_ = open(path, O_RDWR);
		if(fd_ < 0) {
			rt_fprintf(stderr, "OLED: failed to open %s\n", path);
			return false;
		}
		if(ioctl(fd_, I2C_SLAVE, address) < 0) {
			rt_fprintf(stderr, "OLED: failed to set I2C address 0x%02X\n", address);
			close(fd_);
			fd_ = -1;
			return false;
		}
		clear();
		if(!initSequence()) {
			rt_fprintf(stderr, "OLED: init sequence failed\n");
			cleanup();
			return false;
		}
		display();
		return true;
	}

	void cleanup()
	{
		if(fd_ >= 0) {
			close(fd_);
			fd_ = -1;
		}
	}

	void clear()
	{
		memset(buffer_, 0, sizeof(buffer_));
	}

	void pixel(int x, int y, bool on = true)
	{
		if(x < 0 || x >= kOledWidth || y < 0 || y >= kOledHeight)
			return;
		int page = y / 8;
		int index = x + page * kOledWidth;
		uint8_t mask = 1 << (y % 8);
		if(on)
			buffer_[index] |= mask;
		else
			buffer_[index] &= ~mask;
	}

	void hLine(int x0, int x1, int y, bool on = true)
	{
		if(y < 0 || y >= kOledHeight)
			return;
		if(x0 > x1)
			std::swap(x0, x1);
		x0 = std::max(0, x0);
		x1 = std::min(kOledWidth - 1, x1);
		for(int x = x0; x <= x1; ++x)
			pixel(x, y, on);
	}

	void rect(int x, int y, int w, int h, bool on = true)
	{
		hLine(x, x + w - 1, y, on);
		hLine(x, x + w - 1, y + h - 1, on);
		for(int yy = y; yy < y + h; ++yy) {
			pixel(x, yy, on);
			pixel(x + w - 1, yy, on);
		}
	}

	void fillRect(int x, int y, int w, int h, bool on = true)
	{
		for(int yy = y; yy < y + h; ++yy)
			for(int xx = x; xx < x + w; ++xx)
				pixel(xx, yy, on);
	}

	int glyphWidth(char c)
	{
		char n = normalizeChar(c);
		return (n == '{' || n == '}' || n == '~' || n == '@') ? 7 : 5;
	}

	void drawChar(int x, int y, char c, bool on = true)
	{
		char n = normalizeChar(c);
		const uint8_t* glyph = glyphFor(n);
		if(!glyph)
			return;
		int w = glyphWidth(n);
		for(int col = 0; col < w; ++col) {
			uint8_t bits = glyph[col];
			for(int row = 0; row < 7; ++row) {
				if(bits & (1 << row))
					pixel(x + col, y + row, on);
			}
		}
	}

	void drawText(int x, int y, const std::string& s, bool on = true)
	{
		int cursor = x;
		for(char c : s) {
			if(c == '\n') {
				y += 8;
				cursor = x;
				continue;
			}
			char n = normalizeChar(c);
			drawChar(cursor, y, n, on);
			cursor += glyphWidth(n) + 1;
		}
	}

	int textWidth(const std::string& s)
	{
		int w = 0;
		for(char c : s) {
			if(c == '\n')
				break;
			char n = normalizeChar(c);
			w += glyphWidth(n) + 1;
		}
		return std::max(0, w - 1);
	}

	void display()
	{
		if(fd_ < 0)
			return;
		for(int page = 0; page < 8; ++page) {
			sendCommand(0xB0 + page);
			sendCommand(0x00);
			sendCommand(0x10);

			uint8_t out[1 + kOledWidth];
			out[0] = 0x40;
			memcpy(&out[1], &buffer_[page * kOledWidth], kOledWidth);
			if(write(fd_, out, sizeof(out)) != (ssize_t)sizeof(out)) {
				rt_fprintf(stderr, "OLED: page write failed\n");
				return;
			}
		}
	}

private:
	int fd_ = -1;
	uint8_t buffer_[kOledWidth * (kOledHeight / 8)]{};

	static char normalizeChar(char c)
	{
		return c;
	}

	bool sendCommand(uint8_t cmd)
	{
		uint8_t out[2] = {0x00, cmd};
		return write(fd_, out, 2) == 2;
	}

	bool initSequence()
	{
		const uint8_t init[] = {
			0xAE,
			0xD5, 0x80,
			0xA8, 0x3F,
			0xD3, 0x00,
			0x40,
			0x8D, 0x14,
			0x20, 0x00,
			0xA1,
			0xC8,
			0xDA, 0x12,
			0x81, 0x7F,
			0xD9, 0xF1,
			0xDB, 0x40,
			0xA4,
			0xA6,
			0x2E,
			0xAF
		};
		for(size_t i = 0; i < sizeof(init); ++i) {
			if(!sendCommand(init[i]))
				return false;
		}
		return true;
	}

	const uint8_t* glyphFor(char c)
	{
		static const uint8_t space[5] = {0,0,0,0,0};
		static const uint8_t dash[5]  = {0x08,0x08,0x08,0x08,0x08};
		static const uint8_t dot[5]   = {0x00,0x60,0x60,0x00,0x00};
		static const uint8_t comma[5] = {0x00,0x50,0x30,0x00,0x00};
		static const uint8_t colon[5] = {0x00,0x36,0x36,0x00,0x00};
		static const uint8_t plus[5]  = {0x08,0x08,0x3E,0x08,0x08};
		static const uint8_t hash[5]  = {0x14,0x3E,0x14,0x3E,0x14};
		static const uint8_t dollar[5]= {0x24,0x2A,0x7F,0x2A,0x12};
		static const uint8_t percent[5]= {0x23,0x13,0x08,0x64,0x62};
		static const uint8_t amp[5]   = {0x36,0x49,0x55,0x22,0x50};
		static const uint8_t slash[5] = {0x20,0x10,0x08,0x04,0x02};
		static const uint8_t lpar[5]  = {0x00,0x1C,0x22,0x41,0x00};
		static const uint8_t rpar[5]  = {0x00,0x41,0x22,0x1C,0x00};
		static const uint8_t under[5] = {0x40,0x40,0x40,0x40,0x40};
		static const uint8_t equal[5] = {0x14,0x14,0x14,0x14,0x14};
		static const uint8_t excl[5]  = {0x00,0x00,0x5F,0x00,0x00};
		static const uint8_t quest[5] = {0x02,0x01,0x51,0x09,0x06};
		// Custom preset symbols (7x7) from user pixel-art.
		// '{' = heart, '}' = star, '~' = note, '@' = smile
		static const uint8_t heart[7] = {0x0E,0x11,0x21,0x42,0x21,0x11,0x0E};
		static const uint8_t smile[7] = {0x1C,0x2A,0x55,0x51,0x55,0x2A,0x1C};
		static const uint8_t note[7]  = {0x02,0x03,0x3F,0x78,0x78,0x30,0x00};
		static const uint8_t star[7]  = {0x2C,0x54,0x42,0x21,0x42,0x54,0x2C};

		static const uint8_t n0[5] = {0x3E,0x51,0x49,0x45,0x3E};
		static const uint8_t n1[5] = {0x00,0x42,0x7F,0x40,0x00};
		static const uint8_t n2[5] = {0x42,0x61,0x51,0x49,0x46};
		static const uint8_t n3[5] = {0x21,0x41,0x45,0x4B,0x31};
		static const uint8_t n4[5] = {0x18,0x14,0x12,0x7F,0x10};
		static const uint8_t n5[5] = {0x27,0x45,0x45,0x45,0x39};
		static const uint8_t n6[5] = {0x3C,0x4A,0x49,0x49,0x30};
		static const uint8_t n7[5] = {0x01,0x71,0x09,0x05,0x03};
		static const uint8_t n8[5] = {0x36,0x49,0x49,0x49,0x36};
		static const uint8_t n9[5] = {0x06,0x49,0x49,0x29,0x1E};

		static const uint8_t A[5] = {0x7E,0x11,0x11,0x11,0x7E};
		static const uint8_t B[5] = {0x7F,0x49,0x49,0x49,0x36};
		static const uint8_t C[5] = {0x3E,0x41,0x41,0x41,0x22};
		static const uint8_t D[5] = {0x7F,0x41,0x41,0x22,0x1C};
		static const uint8_t E[5] = {0x7F,0x49,0x49,0x49,0x41};
		static const uint8_t F[5] = {0x7F,0x09,0x09,0x09,0x01};
		static const uint8_t G[5] = {0x3E,0x41,0x49,0x49,0x7A};
		static const uint8_t H[5] = {0x7F,0x08,0x08,0x08,0x7F};
		static const uint8_t I[5] = {0x00,0x41,0x7F,0x41,0x00};
		static const uint8_t J[5] = {0x20,0x40,0x41,0x3F,0x01};
		static const uint8_t K[5] = {0x7F,0x08,0x14,0x22,0x41};
		static const uint8_t L[5] = {0x7F,0x40,0x40,0x40,0x40};
		static const uint8_t M[5] = {0x7F,0x02,0x0C,0x02,0x7F};
		static const uint8_t N[5] = {0x7F,0x04,0x08,0x10,0x7F};
		static const uint8_t O[5] = {0x3E,0x41,0x41,0x41,0x3E};
		static const uint8_t P[5] = {0x7F,0x09,0x09,0x09,0x06};
		static const uint8_t Q[5] = {0x3E,0x41,0x51,0x21,0x5E};
		static const uint8_t R[5] = {0x7F,0x09,0x19,0x29,0x46};
		static const uint8_t S[5] = {0x46,0x49,0x49,0x49,0x31};
		static const uint8_t T[5] = {0x01,0x01,0x7F,0x01,0x01};
		static const uint8_t U[5] = {0x3F,0x40,0x40,0x40,0x3F};
		static const uint8_t V[5] = {0x1F,0x20,0x40,0x20,0x1F};
		static const uint8_t W[5] = {0x7F,0x20,0x18,0x20,0x7F};
		static const uint8_t X[5] = {0x63,0x14,0x08,0x14,0x63};
		static const uint8_t Y[5] = {0x03,0x04,0x78,0x04,0x03};
		static const uint8_t Z[5] = {0x61,0x51,0x49,0x45,0x43};

		// Distinct lowercase glyphs for preset names.
		static const uint8_t a_[5] = {0x20,0x54,0x54,0x54,0x78};
		static const uint8_t b_[5] = {0x7F,0x48,0x44,0x44,0x38};
		static const uint8_t c_[5] = {0x38,0x44,0x44,0x44,0x20};
		static const uint8_t d_[5] = {0x38,0x44,0x44,0x48,0x7F};
		static const uint8_t e_[5] = {0x38,0x54,0x54,0x54,0x18};
		static const uint8_t f_[5] = {0x08,0x7E,0x09,0x01,0x02};
		static const uint8_t g_[5] = {0x0C,0x52,0x52,0x52,0x3E};
		static const uint8_t h_[5] = {0x7F,0x08,0x04,0x04,0x78};
		static const uint8_t i_[5] = {0x00,0x44,0x7D,0x40,0x00};
		static const uint8_t j_[5] = {0x20,0x40,0x44,0x3D,0x00};
		static const uint8_t k_[5] = {0x7F,0x10,0x28,0x44,0x00};
		static const uint8_t l_[5] = {0x00,0x41,0x7F,0x40,0x00};
		static const uint8_t m_[5] = {0x7C,0x04,0x18,0x04,0x78};
		static const uint8_t n_[5] = {0x7C,0x08,0x04,0x04,0x78};
		static const uint8_t o_[5] = {0x38,0x44,0x44,0x44,0x38};
		static const uint8_t p_[5] = {0x7C,0x14,0x14,0x14,0x08};
		static const uint8_t q_[5] = {0x08,0x14,0x14,0x18,0x7C};
		static const uint8_t r_[5] = {0x7C,0x08,0x04,0x04,0x08};
		static const uint8_t s_[5] = {0x48,0x54,0x54,0x54,0x20};
		static const uint8_t t_[5] = {0x04,0x3F,0x44,0x40,0x20};
		static const uint8_t u_[5] = {0x3C,0x40,0x40,0x20,0x7C};
		static const uint8_t v_[5] = {0x1C,0x20,0x40,0x20,0x1C};
		static const uint8_t w_[5] = {0x3C,0x40,0x30,0x40,0x3C};
		static const uint8_t x_[5] = {0x44,0x28,0x10,0x28,0x44};
		static const uint8_t y_[5] = {0x0C,0x50,0x50,0x50,0x3C};
		static const uint8_t z_[5] = {0x44,0x64,0x54,0x4C,0x44};

		switch(c) {
			case ' ': return space;
			case '-': return dash;
			case '.': return dot;
			case ',': return comma;
			case ':': return colon;
			case '+': return plus;
			case '#': return hash;
			case '$': return dollar;
			case '%': return percent;
			case '&': return amp;
			case '/': return slash;
			case '(': return lpar;
			case ')': return rpar;
			case '_': return under;
			case '=': return equal;
			case '!': return excl;
			case '?': return quest;
			case '{': return heart;
			case '}': return star;
			case '~': return note;
			case '@': return smile;
			case '0': return n0; case '1': return n1; case '2': return n2; case '3': return n3; case '4': return n4;
			case '5': return n5; case '6': return n6; case '7': return n7; case '8': return n8; case '9': return n9;
			case 'A': return A; case 'a': return a_;
			case 'B': return B; case 'b': return b_;
			case 'C': return C; case 'c': return c_;
			case 'D': return D; case 'd': return d_;
			case 'E': return E; case 'e': return e_;
			case 'F': return F; case 'f': return f_;
			case 'G': return G; case 'g': return g_;
			case 'H': return H; case 'h': return h_;
			case 'I': return I; case 'i': return i_;
			case 'J': return J; case 'j': return j_;
			case 'K': return K; case 'k': return k_;
			case 'L': return L; case 'l': return l_;
			case 'M': return M; case 'm': return m_;
			case 'N': return N; case 'n': return n_;
			case 'O': return O; case 'o': return o_;
			case 'P': return P; case 'p': return p_;
			case 'Q': return Q; case 'q': return q_;
			case 'R': return R; case 'r': return r_;
			case 'S': return S; case 's': return s_;
			case 'T': return T; case 't': return t_;
			case 'U': return U; case 'u': return u_;
			case 'V': return V; case 'v': return v_;
			case 'W': return W; case 'w': return w_;
			case 'X': return X; case 'x': return x_;
			case 'Y': return Y; case 'y': return y_;
			case 'Z': return Z; case 'z': return z_;
			default: return space;
		}
	}
};

static SSD1306 gOled;

// ------------------------------------------------------------
// UI helpers
// ------------------------------------------------------------

static std::string centeredTrim(const std::string& s, int maxChars)
{
	if((int)s.size() <= maxChars)
		return s;
	if(maxChars < 4)
		return s.substr(0, std::max(0, maxChars));
	return s.substr(0, maxChars - 3) + "...";
}

static std::string fullParamName(int page, int idx)
{
	switch(page) {
		case kPagePlay: {
			static const char* n[] = {"Master", "Exciter Blend", "Brightness", "Excitation Position", "Pickup Position", "Wet/Dry"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPagePlayAlt: {
			static const char* n[] = {"Partials", "Density", "Stretch", "Warp", "Dispersion", "Unused"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageBodyA1:
		case kPageBodyB1: {
			static const char* n[] = {"Density", "Stretch", "Warp", "Dispersion", "Brightness", "Partials"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageBodyA2:
		case kPageBodyB2: {
			static const char* n[] = {"Odd Skew", "Even Skew", "Collision", "Release Amount", "Odd/Even Bias", "Unused"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageDampers: {
			static const char* n[] = {"Bell Frequency", "Bell Zeta", "Left Power", "Right Power", "Bell Model", "Unused"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageExciterA: {
			static const char* n[] = {"Exciter Blend", "Attack", "Decay", "Sustain", "Release", "Noise Color"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageExciterB: {
			static const char* n[] = {"Impulse Shape", "Attack Curve", "Decay Curve", "Release Curve", "Unused", "Unused"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageSpace: {
			static const char* n[] = {"Space Size", "Space Decay", "Diffusion", "Damping", "Onset", "Wet/Dry"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageEcho: {
			static const char* n[] = {"Grain Size", "Density", "Spray", "Pitch", "Shape", "Feedback"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageSaturation: {
			static const char* n[] = {"Drive", "Threshold", "Curve", "Asymmetry", "Tone", "Wet/Dry"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageModLfo1:
		case kPageModLfo2: {
			static const char* n[] = {"Target Bank", "Target", "Shape", "Rate", "Mode", "Amount"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageVelocity: {
			static const char* n[] = {"Target Bank", "Target", "Velocity Amount", "Unused", "Unused", "Unused"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPagePressure: {
			static const char* n[] = {"Target Bank", "Target", "Pressure Amount", "Dead Zone", "Curve", "Unused"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageGlobalEdit: {
			static const char* n[] = {"Bank Select", "Octave", "Semitone", "Tune", "Partials", "Unused"};
			return n[std::max(0, std::min(idx, 5))];
		}
		case kPageResonatorEdit: {
			static const char* n[] = {"Resonator Index", "Ratio", "Gain", "Decay", "Unused", "Unused"};
			return n[std::max(0, std::min(idx, 5))];
		}
		default:
			return "Parameter";
	}
}

static const char* pageName(int page)
{
	switch(page) {
		case kPagePlay: return "PLAY";
		case kPagePlayAlt: return "PLAY ALT";
		case kPageBodyA1: return "BODY A1";
		case kPageBodyA2: return "BODY A2";
		case kPageBodyB1: return "BODY B1";
		case kPageBodyB2: return "BODY B2";
		case kPageDampers: return "DAMPERS";
		case kPageExciterA: return "EXC A";
		case kPageExciterB: return "EXC B";
		case kPageSpace: return "SPACE";
		case kPageEcho: return "ECHO";
		case kPageSaturation: return "SAT";
		case kPageModLfo1: return "LFO1";
		case kPageModLfo2: return "LFO2";
		case kPageVelocity: return "VELOCITY";
		case kPagePressure: return "PRESSURE";
		case kPageGlobalEdit: return "GLOBAL";
		case kPageResonatorEdit: return "RES EDIT";
		case kPagePreset: return "PRESET";
		default: return "PAGE";
	}
}

static void pageLabels(int page, const char* labels[kParamCount])
{
	static const char* blank[kParamCount]  = {"---","---","---","---","---","---"};
	static const char* play[kParamCount]   = {"MSTR","EXC","BRGT","POS","PICK","WET"};
	static const char* playAlt[kParamCount]= {"PART","DENS","STR","WARP","DISP","---"};
	static const char* body1[kParamCount]  = {"DENS","STR","WARP","DISP","BRGT","PART"};
	static const char* body2[kParamCount]  = {"ODDSK","EVNSK","COLL","RELAM","OEBIA","---"};
	static const char* damp[kParamCount]   = {"FREQ","ZETA","LPOW","RPOW","MODEL","---"};
	static const char* excA[kParamCount]   = {"EXC","ATK","DEC","SUS","REL","NCOL"};
	static const char* excB[kParamCount]   = {"IMPL","ATKC","DECC","RELC","---","---"};
	static const char* space[kParamCount]  = {"SIZE","DEC","DIFF","DAMP","ONST","WET"};
	static const char* echo[kParamCount]   = {"SIZE","DENS","SPRY","PITC","SHAP","FDBK"};
	static const char* sat[kParamCount]    = {"DRIV","THR","CURV","ASYM","TONE","WET"};
	static const char* lfo[kParamCount]    = {"BANK","TGT","SHAPE","RATE","MODE","AMT"};
	static const char* vel[kParamCount]    = {"BANK","TGT","VELA","---","---","---"};
	static const char* press[kParamCount]  = {"BANK","TGT","PAMT","DZ","CURV","---"};
	static const char* glob[kParamCount]   = {"BANK","OCTV","SEMI","TUNE","PART","---"};
	static const char* res[kParamCount]    = {"RIDX","RAT","GAIN","DECAY","---","---"};
	static const char* preset[kParamCount] = {"SLOT","MODE","USED","CUR","---","---"};

	switch(page) {
		case kPagePlay:       memcpy(labels, play, sizeof(play)); break;
		case kPagePlayAlt:    memcpy(labels, playAlt, sizeof(playAlt)); break;
		case kPageBodyA1:
		case kPageBodyB1:      memcpy(labels, body1, sizeof(body1)); break;
		case kPageBodyA2:
		case kPageBodyB2:      memcpy(labels, body2, sizeof(body2)); break;
		case kPageDampers:    memcpy(labels, damp, sizeof(damp)); break;
		case kPageExciterA:   memcpy(labels, excA, sizeof(excA)); break;
		case kPageExciterB:   memcpy(labels, excB, sizeof(excB)); break;
		case kPageSpace:      memcpy(labels, space, sizeof(space)); break;
		case kPageEcho:       memcpy(labels, echo, sizeof(echo)); break;
		case kPageSaturation: memcpy(labels, sat, sizeof(sat)); break;
		case kPageModLfo1:
		case kPageModLfo2:    memcpy(labels, lfo, sizeof(lfo)); break;
		case kPageVelocity:   memcpy(labels, vel, sizeof(vel)); break;
		case kPagePressure:   memcpy(labels, press, sizeof(press)); break;
		case kPageGlobalEdit: memcpy(labels, glob, sizeof(glob)); break;
		case kPageResonatorEdit: memcpy(labels, res, sizeof(res)); break;
		case kPagePreset:     memcpy(labels, preset, sizeof(preset)); break;
		default:              memcpy(labels, blank, sizeof(blank)); break;
	}
}


static std::string formatFloatSmart(float v, int maxDecimals = 2)
{
	char buf[32];
	float av = std::fabs(v);
	if(av < 0.0005f) v = 0.f;
	if(std::fabs(v - std::round(v)) < 0.0005f) {
		snprintf(buf, sizeof(buf), "%.0f", v);
		return buf;
	}
	if(maxDecimals <= 1 || std::fabs(v * 10.f - std::round(v * 10.f)) < 0.0005f) {
		snprintf(buf, sizeof(buf), "%.1f", v);
		return buf;
	}
	snprintf(buf, sizeof(buf), "%.2f", v);
	return buf;
}

static inline float displayDensityValue(float v)
{
	/* Density now arrives from the synth already in its public domain:
	   0..1 = collapse side, 1..6 = widening side, with 1 = normal spacing. */
	return std::max(0.f, std::min(6.f, v));
}

static inline float displayStretchValue(float v)
{
	return std::max(0.f, std::min(2.f, 1.f + v));
}

static inline float displayWarpValue(float v)
{
	return std::pow(2.f, std::max(-1.f, std::min(1.f, v)) * 2.f);
}

static inline float displaySkewValue(float v)
{
	return std::pow(2.f, std::max(-1.f, std::min(1.f, v)));
}

static inline float displayBrightnessValue(float v)
{
	return 1.f - std::max(-1.f, std::min(1.f, v));
}

static std::string formatValueCore(int page, int idx, float v, bool overlay)
{
	char buf[32];
	if(std::fabs(v) < 0.0005f)
		v = 0.0f;

	// selectors / categorical
	if(page == kPagePreset && idx == 0) {
		snprintf(buf, sizeof(buf), "%d", (int)std::round(v) + 1);
		return buf;
	}
	if(page == kPagePreset && idx == 1) {
		int m = (int)std::round(v);
		if(m <= 0) return "LOAD";
		if(m == 1) return "NAME";
		return "SAVE";
	}
	if(page == kPagePreset && idx == 2)
		return (v > 0.5f) ? "USED" : "EMPTY";
	if(page == kPagePreset && idx == 3) {
		snprintf(buf, sizeof(buf), "%d", (int)std::round(v) + 1);
		return buf;
	}
	if((page == kPageModLfo1 || page == kPageModLfo2) && idx == 0) {
		int m = (int)std::round(v);
		if(m <= 1) return "A";
		if(m == 2) return "B";
		return "AB";
	}
	if((page == kPageModLfo1 || page == kPageModLfo2) && idx == 1) {
		static const char* names[] = {"NONE","MSTR","PITC","BRGT","POS","PICK","PART","IMPL","NCOL","L2RT","L2AM"};
		return names[std::max(0, std::min((int)std::round(v), 10))];
	}
	if((page == kPageModLfo1 || page == kPageModLfo2) && idx == 2) {
		static const char* names[] = {"SIN","SAW","SQR","RND","SDN"};
		return names[std::max(0, std::min((int)std::round(v) - 1, 4))];
	}
	if((page == kPageModLfo1 || page == kPageModLfo2) && idx == 4) {
		int m = (int)std::round(v);
		if(m <= 1) return "FREE";
		return "SHOT";
	}
	if(page == kPageVelocity && idx == 0) {
		int m = (int)std::round(v);
		if(m <= 1) return "A";
		if(m == 2) return "B";
		return "AB";
	}
	if(page == kPageVelocity && idx == 1) {
		static const char* names[] = {"NONE","MSTR","BRGT","POS","PICK","ATK","DEC","REL","IMPL","NCOL","D1","D2","D3"};
		return names[std::max(0, std::min((int)std::round(v), 12))];
	}
	if(page == kPagePressure && idx == 0) {
		int m = (int)std::round(v);
		if(m <= 1) return "A";
		if(m == 2) return "B";
		return "AB";
	}
	if(page == kPagePressure && idx == 1) {
		static const char* names[] = {"NONE","MSTR","BRGT","POS","PICK","ATK","DEC","REL","IMPL","NCOL","D1","D2","D3"};
		return names[std::max(0, std::min((int)std::round(v), 12))];
	}

	// integer/count displays
	if(page == kPagePlayAlt && idx == 0) return formatFloatSmart(std::round(v), 0);
	if((page == kPageBodyA1 || page == kPageBodyB1 || page == kPageGlobalEdit) && idx == 5) return formatFloatSmart(std::round(v), 0);
	if(page == kPageGlobalEdit && idx == 0) {
		int m = (int)std::round(v);
		if(m <= 1) return "A";
		if(m == 2) return "B";
		return "AB";
	}
	if(page == kPageGlobalEdit && idx == 1) {
		snprintf(buf, sizeof(buf), overlay ? "%+d oct" : "%+d", (int)std::round(v));
		return buf;
	}
	if(page == kPageGlobalEdit && idx == 2) {
		snprintf(buf, sizeof(buf), overlay ? "%+d st" : "%+d", (int)std::round(v));
		return buf;
	}
	if(page == kPageGlobalEdit && idx == 3) {
		snprintf(buf, sizeof(buf), overlay ? "%+d ct" : "%+d", (int)std::round(v));
		return buf;
	}
	if(page == kPageGlobalEdit && idx == 4) return formatFloatSmart(std::round(v), 0);
	if(page == kPageResonatorEdit && idx == 0) return formatFloatSmart(std::round(v), 0);

	// exact/frequency/time displays
	if(page == kPageDampers && idx == 0) {
		float hz = v;
		if(overlay) snprintf(buf, sizeof(buf), "%.0fHz", hz);
		else snprintf(buf, sizeof(buf), "%.0f", hz);
		return buf;
	}
	if((page == kPageExciterA && (idx == 1 || idx == 2 || idx == 4)) ||
	   (page == kPageResonatorEdit && idx == 3)) {
		if(overlay) snprintf(buf, sizeof(buf), "%.0fms", v);
		else snprintf(buf, sizeof(buf), "%.0f", v);
		return buf;
	}
	if((page == kPageModLfo1 || page == kPageModLfo2) && idx == 3) {
		if(overlay) snprintf(buf, sizeof(buf), "%.2fHz", v);
		else snprintf(buf, sizeof(buf), "%.2f", v);
		return buf;
	}
	if((page == kPageModLfo1 || page == kPageModLfo2) && idx == 5) {
		return formatFloatSmart(v, 2);
	}
	if(page == kPagePlayAlt && idx == 1) {
		float d = displayDensityValue(v);
		return formatFloatSmart(d, 2);
	}
	if((page == kPageBodyA1 || page == kPageBodyB1) && idx == 0) {
		float d = displayDensityValue(v);
		return formatFloatSmart(d, 2);
	}
	if(page == kPagePlayAlt && idx == 2) {
		float m = displayStretchValue(v);
		std::string s = formatFloatSmart(m, 2);
		return overlay ? (s + "x") : s;
	}
	if((page == kPageBodyA1 || page == kPageBodyB1) && idx == 1) {
		float m = displayStretchValue(v);
		std::string s = formatFloatSmart(m, 2);
		return overlay ? (s + "x") : s;
	}
	if(page == kPagePlayAlt && idx == 3) {
		float p = displayWarpValue(v);
		std::string s = formatFloatSmart(p, 2);
		return overlay ? (s + "p") : s;
	}
	if((page == kPageBodyA1 || page == kPageBodyB1) && idx == 2) {
		float p = displayWarpValue(v);
		std::string s = formatFloatSmart(p, 2);
		return overlay ? (s + "p") : s;
	}
	if((page == kPageBodyA1 || page == kPageBodyB1) && idx == 4) {
		float a = displayBrightnessValue(v);
		return formatFloatSmart(a, 2);
	}
	if((page == kPageBodyA2 || page == kPageBodyB2) && idx == 0) {
		float m = displaySkewValue(v);
		std::string s = formatFloatSmart(m, 2);
		return overlay ? (s + "x") : s;
	}
	if((page == kPageBodyA2 || page == kPageBodyB2) && idx == 1) {
		float m = displaySkewValue(v);
		std::string s = formatFloatSmart(m, 2);
		return overlay ? (s + "x") : s;
	}
	if(page == kPageResonatorEdit && idx == 1) {
		std::string s = formatFloatSmart(v, 2);
		return overlay ? (s + "x") : s;
	}
	if(page == kPageResonatorEdit && idx == 2) {
		return formatFloatSmart(v, 2);
	}
	if((page == kPagePlay || page == kPageSpace || page == kPageSaturation) && idx == 5) {
		float wet = std::max(0.f, std::min(100.f, (v + 1.f) * 50.f));
		float dry = 100.f - wet;
		if(overlay) {
			if(v >= 0.f) snprintf(buf, sizeof(buf), "W%.0f D%.0f", wet, dry);
			else        snprintf(buf, sizeof(buf), "D%.0f W%.0f", dry, wet);
			return buf;
		}
		return formatFloatSmart((v >= 0.f) ? wet : dry, 0);
	}
	if(page == kPagePlay && idx == 1) {
		float bAmt = std::max(0.f, std::min(100.f, (v + 1.f) * 50.f));
		float aAmt = 100.f - bAmt;
		if(overlay) {
			if(v >= 0.f) snprintf(buf, sizeof(buf), "B%.0f A%.0f", bAmt, aAmt);
			else        snprintf(buf, sizeof(buf), "A%.0f B%.0f", aAmt, bAmt);
			return buf;
		}
		return formatFloatSmart((v >= 0.f) ? bAmt : aAmt, 0);
	}
	if((page == kPageModLfo1 || page == kPageModLfo2) && idx == 4) return overlay ? std::string(((int)std::round(v) <= 1) ? "FREE" : "SHOT") : std::string(((int)std::round(v) <= 1) ? "FREE" : "SHOT");
	if((page == kPageEcho && idx == 3)) return formatFloatSmart(v, 2);

	const bool bipolar =
		(page == kPagePlay && (idx == 2)) ||
		(page == kPageExciterA && idx == 0) ||
		(page == kPageVelocity && idx == 2) ||
		(page == kPagePressure && (idx == 2 || idx == 4)) ||
		(page == kPageSaturation && (idx == 3 || idx == 4));

	if(bipolar) {
		snprintf(buf, sizeof(buf), "%+.2f", v);
		return buf;
	}

	return formatFloatSmart(v, 2);
}

static std::string formatValue(int page, int idx, float v)
{
	return formatValueCore(page, idx, v, false);
}

static std::string formatValueOverlay(int page, int idx, float v)
{
	return formatValueCore(page, idx, v, true);
}

static bool isOverlayEligiblePage(int page)
{
	return page != kPagePreset;
}

static void drawCenteredText(int y, const std::string& s, bool on = true)
{
	int w = gOled.textWidth(s);
	int x = std::max(0, (kOledWidth - w) / 2);
	gOled.drawText(x, y, s, on);
}

static inline void noteParamDisplayChangeLocked(int idx, float oldv, float newv)
{
	if(idx < 0 || idx >= kParamCount)
		return;
	uint64_t nowMs = uiNowMs();
	if(nowMs < gIgnoreParamUntilMs)
		return;
	int page = gScreenState.page;
	if(!isOverlayEligiblePage(page))
		return;
	if(formatValueOverlay(page, idx, oldv) != formatValueOverlay(page, idx, newv))
		gParamChangedMs[idx] = nowMs;
}


static inline bool isBodyPage(int page)
{
	return page == kPageBodyA1 || page == kPageBodyA2 || page == kPageBodyB1 || page == kPageBodyB2;
}

static inline float norm01(float v, float lo, float hi)
{
	if(hi <= lo) return 0.f;
	float t = (v - lo) / (hi - lo);
	return std::max(0.f, std::min(1.f, t));
}

static inline float norm01Bipolar(float v)
{
	return norm01(v, -1.f, 1.f);
}

static inline float normDamperPositive(float v, float hi)
{
	if(v <= 1.05f)
		return std::max(0.f, std::min(1.f, v));
	return norm01(v, 0.f, hi);
}

static inline float normDamperFreq(float v)
{
	if(v <= 1.05f)
		return std::max(0.f, std::min(1.f, v));
	const float lo = 20.f;
	const float hi = 16000.f;
	float hz = std::max(lo, std::min(hi, v));
	float a = std::log(hz / lo);
	float b = std::log(hi / lo);
	if(b <= 0.f)
		return 0.f;
	return std::max(0.f, std::min(1.f, a / b));
}

static inline float normDamperModel(float v)
{
	if(v <= 1.05f)
		return std::max(0.f, std::min(1.f, v));
	return norm01(v, 0.f, 8.f);
}

static inline void clippedPixelSet(int boxX, int boxY, int boxW, int boxH, int x, int y, bool on)
{
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int ix1 = boxX + boxW - 2;
	const int iy1 = boxY + boxH - 2;
	if(x < ix0 || x > ix1 || y < iy0 || y > iy1)
		return;
	gOled.pixel(x, y, on);
}

static inline void clippedPixel(int boxX, int boxY, int boxW, int boxH, int x, int y)
{
	clippedPixelSet(boxX, boxY, boxW, boxH, x, y, true);
}

static inline void clippedDot(int boxX, int boxY, int boxW, int boxH, int x, int y, int r = 0)
{
	for(int yy = y - r; yy <= y + r; ++yy)
		for(int xx = x - r; xx <= x + r; ++xx)
			clippedPixel(boxX, boxY, boxW, boxH, xx, yy);
}

static inline void clippedSoftBlob(int boxX, int boxY, int boxW, int boxH, int x, int y, float amt, int seed = 0)
{
	amt = std::max(0.f, std::min(1.f, amt));
	if(amt <= 0.f)
		return;
	clippedPixel(boxX, boxY, boxW, boxH, x, y);
	if(amt > 0.18f) clippedPixel(boxX, boxY, boxW, boxH, x + 1, y);
	if(amt > 0.30f) clippedPixel(boxX, boxY, boxW, boxH, x - 1, y);
	if(amt > 0.44f) clippedPixel(boxX, boxY, boxW, boxH, x, y + 1);
	if(amt > 0.58f) clippedPixel(boxX, boxY, boxW, boxH, x, y - 1);
	if(amt > 0.72f && ((seed + x + y) & 1) == 0) clippedPixel(boxX, boxY, boxW, boxH, x + 1, y + 1);
	if(amt > 0.80f && ((seed + x) & 1) == 0) clippedPixel(boxX, boxY, boxW, boxH, x - 1, y - 1);
	if(amt > 0.88f && ((seed + y) & 1) == 0) clippedPixel(boxX, boxY, boxW, boxH, x + 1, y - 1);
	if(amt > 0.96f) clippedPixel(boxX, boxY, boxW, boxH, x - 1, y + 1);
}

static void clippedLine(int boxX, int boxY, int boxW, int boxH, float x0, float y0, float x1, float y1, int thick = 0)
{
	int steps = (int)std::max(std::fabs(x1 - x0), std::fabs(y1 - y0));
	if(steps < 1) steps = 1;
	for(int i = 0; i <= steps; ++i) {
		float t = (float)i / (float)steps;
		int x = (int)std::lround(x0 + (x1 - x0) * t);
		int y = (int)std::lround(y0 + (y1 - y0) * t);
		clippedDot(boxX, boxY, boxW, boxH, x, y, thick);
	}
}

static void clippedCircle(int boxX, int boxY, int boxW, int boxH, float cx, float cy, float r, int thick = 0, float yScale = 1.f)
{
	if(r <= 0.5f) return;
	const int n = 96;
	for(int i = 0; i < n; ++i) {
		float a0 = (2.f * (float)M_PI * i) / (float)n;
		float a1 = (2.f * (float)M_PI * (i + 1)) / (float)n;
		float x0 = cx + std::cos(a0) * r;
		float y0 = cy + std::sin(a0) * r * yScale;
		float x1 = cx + std::cos(a1) * r;
		float y1 = cy + std::sin(a1) * r * yScale;
		clippedLine(boxX, boxY, boxW, boxH, x0, y0, x1, y1, thick);
	}
}

static void clippedHollowNode(int boxX, int boxY, int boxW, int boxH, int cx, int cy)
{
	static const int pts[][2] = {
		{0,-2}, {1,-2}, {-1,-2},
		{-2,-1}, {2,-1},
		{-2,0}, {2,0},
		{-2,1}, {2,1},
		{-1,2}, {0,2}, {1,2}
	};
	for(const auto& p : pts)
		clippedPixel(boxX, boxY, boxW, boxH, cx + p[0], cy + p[1]);
	for(int yy = -1; yy <= 1; ++yy)
		for(int xx = -1; xx <= 1; ++xx)
			clippedPixelSet(boxX, boxY, boxW, boxH, cx + xx, cy + yy, false);
	clippedPixelSet(boxX, boxY, boxW, boxH, cx, cy, false);
}


static inline void clippedMaskDot(int boxX, int boxY, int boxW, int boxH, int x, int y, int r = 1)
{
	for(int yy = y - r; yy <= y + r; ++yy)
		for(int xx = x - r; xx <= x + r; ++xx)
			clippedPixelSet(boxX, boxY, boxW, boxH, xx, yy, false);
}

static inline void knockedOutDot(int boxX, int boxY, int boxW, int boxH, int x, int y, int r = 0, int halo = 1)
{
	clippedMaskDot(boxX, boxY, boxW, boxH, x, y, r + halo);
	clippedDot(boxX, boxY, boxW, boxH, x, y, r);
}

static inline void knockedOutPixel(int boxX, int boxY, int boxW, int boxH, int x, int y, int halo = 1)
{
	knockedOutDot(boxX, boxY, boxW, boxH, x, y, 0, halo);
}

static void knockedOutLine(int boxX, int boxY, int boxW, int boxH, float x0, float y0, float x1, float y1, int thick = 0, int halo = 1)
{
	int steps = (int)std::max(std::fabs(x1 - x0), std::fabs(y1 - y0));
	if(steps < 1) steps = 1;
	for(int i = 0; i <= steps; ++i) {
		float t = (float)i / (float)steps;
		int x = (int)std::lround(x0 + (x1 - x0) * t);
		int y = (int)std::lround(y0 + (y1 - y0) * t);
		knockedOutDot(boxX, boxY, boxW, boxH, x, y, thick, halo);
	}
}

static void knockedOutSoftBlob(int boxX, int boxY, int boxW, int boxH, int x, int y, float amt, int seed = 0, int halo = 1)
{
	amt = std::max(0.f, std::min(1.f, amt));
	if(amt <= 0.f)
		return;
	clippedMaskDot(boxX, boxY, boxW, boxH, x, y, halo + (amt > 0.55f ? 1 : 0));
	clippedSoftBlob(boxX, boxY, boxW, boxH, x, y, amt, seed);
}



static inline float smoothVisualValue(float& state, float target);

static void drawEchoExactPattern(int boxX, int boxY, int boxW, int boxH,
	int x0, int y0, const int* pts, int count, int patW, int patH, int halo = 1);

static inline float echoHash01(int seed, float salt = 0.f)
{
	return 0.5f + 0.5f * std::sin((float)seed * 12.9898f + salt * 78.233f);
}

static void drawEchoGrain1(int boxX, int boxY, int boxW, int boxH, float cx, float cy, int halo = 1)
{
	static const int pts[] = {0,0};
	int x0 = (int)std::lround(cx);
	int y0 = (int)std::lround(cy);
	drawEchoExactPattern(boxX, boxY, boxW, boxH, x0, y0, pts, 1, 1, 1, halo);
}

static void drawEchoGrain4(int boxX, int boxY, int boxW, int boxH, float cx, float cy, int halo = 1)
{
	int x0 = (int)std::lround(cx) - 1;
	int y0 = (int)std::lround(cy) - 1;
	static const int pts[] = {
		1,0,
		0,1, 2,1,
		1,2
	};
	drawEchoExactPattern(boxX, boxY, boxW, boxH, x0, y0, pts, 4, 3, 3, halo);
}

static void drawEchoGrain8(int boxX, int boxY, int boxW, int boxH, float cx, float cy, int halo = 1)
{
	int x0 = (int)std::lround(cx) - 1;
	int y0 = (int)std::lround(cy) - 1;
	static const int pts[] = {
		1,0, 2,0,
		0,1, 3,1,
		0,2, 3,2,
		1,3, 2,3
	};
	drawEchoExactPattern(boxX, boxY, boxW, boxH, x0, y0, pts, 8, 4, 4, halo);
}

static void drawEchoPacketDiamond(int boxX, int boxY, int boxW, int boxH, float cx, float cy, float s, int halo = 1)
{
	s = std::max(1.5f, s);
	knockedOutLine(boxX, boxY, boxW, boxH, cx, cy - s, cx + s, cy, 0, halo);
	knockedOutLine(boxX, boxY, boxW, boxH, cx + s, cy, cx, cy + s, 0, halo);
	knockedOutLine(boxX, boxY, boxW, boxH, cx, cy + s, cx - s, cy, 0, halo);
	knockedOutLine(boxX, boxY, boxW, boxH, cx - s, cy, cx, cy - s, 0, halo);
	clippedPixelSet(boxX, boxY, boxW, boxH, (int)std::lround(cx), (int)std::lround(cy), false);
}

static void drawEchoPacketShard(int boxX, int boxY, int boxW, int boxH, float cx, float cy, float len, int halo = 1)
{
	len = std::max(3.f, len);
	float half = len * 0.5f;
	knockedOutLine(boxX, boxY, boxW, boxH, cx - half, cy + 1.f, cx + half, cy - 1.f, 0, halo);
	knockedOutLine(boxX, boxY, boxW, boxH, cx - half * 0.55f, cy - 1.f, cx + half * 0.55f, cy - 2.f, 0, halo);
}

static void drawEchoExactPattern(int boxX, int boxY, int boxW, int boxH,
	int x0, int y0, const int* pts, int count, int patW, int patH, int halo)
{
	for(int yy = y0 - halo; yy < y0 + patH + halo; ++yy)
		for(int xx = x0 - halo; xx < x0 + patW + halo; ++xx)
			clippedPixelSet(boxX, boxY, boxW, boxH, xx, yy, false);
	for(int i = 0; i < count; ++i)
		clippedPixel(boxX, boxY, boxW, boxH, x0 + pts[i * 2 + 0], y0 + pts[i * 2 + 1]);
}

static inline float echoLocalGrainState(float grainSize01, int seed)
{
	grainSize01 = std::max(0.f, std::min(1.f, grainSize01));
	float jitter = (echoHash01(seed, grainSize01 * 0.37f) - 0.5f) * 0.22f;
	return std::max(0.f, std::min(1.f, grainSize01 + jitter));
}

static void drawEchoGrainVaried(int boxX, int boxY, int boxW, int boxH, float cx, float cy, float grainSize01, int seed, int halo = 1)
{
	grainSize01 = std::max(0.f, std::min(1.f, grainSize01));
	float local = echoLocalGrainState(grainSize01, seed);
	float pick = echoHash01(seed + 97, local * 0.53f);

	// Global grain-size state:
	// low  = only 1-pixel grains
	// mid  = 1-pixel + 4-pixel grains
	// high = 1-pixel + 4-pixel + 8-pixel grains
	if(grainSize01 < 0.18f) {
		drawEchoGrain1(boxX, boxY, boxW, boxH, cx, cy, halo);
		return;
	}

	if(grainSize01 < 0.55f) {
		float mix4 = std::max(0.f, std::min(1.f, (grainSize01 - 0.18f) / 0.37f));
		float p4 = 0.18f + 0.72f * mix4;
		if(pick < p4 || local > 0.42f)
			drawEchoGrain4(boxX, boxY, boxW, boxH, cx, cy, halo);
		else
			drawEchoGrain1(boxX, boxY, boxW, boxH, cx, cy, halo);
		return;
	}

	float mix8 = std::max(0.f, std::min(1.f, (grainSize01 - 0.55f) / 0.45f));
	float p8 = 0.16f + 0.70f * mix8;
	float p4 = 0.52f - 0.20f * mix8;
	if(local > 0.82f || pick < p8)
		drawEchoGrain8(boxX, boxY, boxW, boxH, cx, cy, halo);
	else if(pick < p8 + p4)
		drawEchoGrain4(boxX, boxY, boxW, boxH, cx, cy, halo);
	else
		drawEchoGrain1(boxX, boxY, boxW, boxH, cx, cy, halo);
}

static void drawEchoPacketMorph(int boxX, int boxY, int boxW, int boxH, float cx, float cy, float grainSize01, float shape, int seed, int halo = 1)
{
	shape = std::max(0.f, std::min(1.f, shape));
	float local = echoLocalGrainState(grainSize01, seed);
	if(shape < 0.33f) {
		drawEchoGrainVaried(boxX, boxY, boxW, boxH, cx, cy, grainSize01, seed, halo);
	} else if(shape < 0.70f) {
		float s = 1.35f + local * 3.6f;
		drawEchoPacketDiamond(boxX, boxY, boxW, boxH, cx, cy, s, halo);
	} else {
		float l = 3.5f + local * 7.0f;
		drawEchoPacketShard(boxX, boxY, boxW, boxH, cx, cy, l, halo);
	}
}

static void drawEchoVisual(int selected, const float values[kParamCount], int boxX, int boxY, int boxW, int boxH)
{
	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;
	float grainSize01 = smoothVisualValue(gEchoSmooth[0], norm01(values[0], 0.f, 1.f));
	float u = (selected == 3) ? norm01Bipolar(values[selected]) : norm01(values[selected], 0.f, 1.f);
	if(selected != 0)
		u = smoothVisualValue(gEchoSmooth[std::max(0, std::min(selected, kParamCount - 1))], u);
	else
		u = grainSize01;

	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float left = ix0 + 6.f;
	const float right = ix0 + iw - 7.f;
	const float cy = iy0 + ih * 0.5f;
	const float usableW = std::max(16.f, right - left);

	auto looseX = [&](float fi, int i, float amt)
	{
		return left + fi * usableW
			+ std::sin(t * 0.85f + i * 1.37f) * amt
			+ std::cos(t * 1.17f + i * 0.61f) * amt * 0.55f;
	};

	auto looseY = [&](float baseY, int i, float amt)
	{
		return baseY
			+ std::sin(t * 1.05f + i * 0.73f) * amt
			+ std::cos(t * 0.72f + i * 1.11f) * amt * 0.55f;
	};

	auto grainPacket = [&](float x, float y, int seed, int halo = 1)
	{
		drawEchoGrainVaried(boxX, boxY, boxW, boxH, x, y, grainSize01, seed, halo);
	};

	auto shapePacket = [&](float x, float y, float shape, int seed, int halo = 1)
	{
		drawEchoPacketMorph(boxX, boxY, boxW, boxH, x, y, grainSize01, shape, seed, halo);
	};

	switch(selected) {
		case 0: { // Grain Size = global state: starts 1px only, then unlocks 4px and 8px grains
			int n = 12 - (int)std::lround(grainSize01 * 3.f);
			n = std::max(8, n);
			for(int i = 0; i < n; ++i) {
				float fi = (float)i / (float)(n - 1);
				float x = looseX(fi, i, 1.2f + grainSize01 * 1.6f);
				float y = looseY(cy, i, 0.55f + grainSize01 * 0.9f);
				grainPacket(x, y, 300 + i, 1);
			}
		} break;

		case 1: { // Density = more grains, sizes controlled by global grain size
			int n = 4 + (int)std::lround(u * 13.f);
			for(int i = 0; i < n; ++i) {
				float fi = (n <= 1) ? 0.5f : (float)i / (float)(n - 1);
				float x = looseX(fi, i, 1.4f + grainSize01 * 2.0f);
				float y = looseY(cy, i, 0.55f + grainSize01 * 0.9f);
				grainPacket(x, y, 400 + i, 1);
			}
		} break;

		case 2: { // Spray = same grain-state, more vertical spread
			const int n = 13;
			float spread = 0.9f + u * (ih * 0.30f);
			for(int i = 0; i < n; ++i) {
				float fi = (float)i / (float)(n - 1);
				float x = looseX(fi, i, 1.6f + grainSize01 * 1.8f);
				float scatter = std::sin(fi * 8.6f + t * 0.8f + i * 0.55f) * spread
					+ std::cos(fi * 5.3f - t * 1.1f + i * 0.41f) * spread * 0.35f;
				float y = looseY(cy + scatter, i, 0.8f + grainSize01 * 1.3f);
				grainPacket(x, y, 500 + i, 1);
			}
		} break;

		case 3: { // Pitch = sloped trail, same global grain-size state
			const int n = 11;
			float slope = u * (ih * 0.22f);
			for(int i = 0; i < n; ++i) {
				float fi = (float)i / (float)(n - 1);
				float x = looseX(fi, i, 1.3f + grainSize01 * 1.6f + std::fabs(u) * 0.6f);
				float y = cy - slope * (fi - 0.5f) * 2.0f;
				y = looseY(y, i, 0.55f + grainSize01 * 0.9f);
				grainPacket(x, y, 600 + i, 1);
			}
		} break;

		case 4: { // Shape = same grain-size state scales the packet forms
			const int n = 8;
			for(int i = 0; i < n; ++i) {
				float fi = (float)i / (float)(n - 1);
				float x = looseX(fi, i, 1.6f + grainSize01 * 1.5f);
				float y = looseY(cy, i, 0.7f + grainSize01 * 1.0f);
				shapePacket(x, y, u, 700 + i, 1);
			}
		} break;

		case 5: { // Feedback = longer surviving chain, sizes still controlled by global grain size
			int visible = 3 + (int)std::lround(u * 10.f);
			const int total = 13;
			for(int i = 0; i < total; ++i) {
				if(i >= visible)
					break;
				float fi = (float)i / (float)(total - 1);
				float x = looseX(fi, i, 0.9f + grainSize01 * 1.4f);
				float y = looseY(cy, i, 0.35f + grainSize01 * 0.55f);
				grainPacket(x, y, 800 + i, 1);
				if(i >= 3) {
					int px = (int)std::lround(x + 2.0f + std::sin(t * 1.3f + i) * 0.8f);
					int py = (int)std::lround(y + std::cos(t * 0.9f + i) * 0.5f);
					knockedOutPixel(boxX, boxY, boxW, boxH, px, py, 1);
				}
			}
		} break;
	}
}

static void drawBodyVisual(int page, int selected, float value, int boxX, int boxY, int boxW, int boxH)
{
	if(!isBodyPage(page))
		return;

	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;

	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	const float maxR = std::min(iw, ih) * 0.42f;

	if(page == kPageBodyA1 || page == kPageBodyB1) {
		switch(selected) {
			case 0: { // Density -> stripe population
				float u = norm01Bipolar(value);
				int stripes = 4 + (int)std::lround(u * 10.f);
				float amp = 1.0f + u * 3.5f;
				for(int i = 0; i < stripes; ++i) {
					float baseY = iy0 + ((i + 1) * (ih - 2)) / (float)(stripes + 1);
					for(int x = ix0; x < ix0 + iw; ++x) {
						float y = baseY
							+ std::sin((x * 0.16f) + t * 2.2f + i * 0.72f) * amp
							+ std::sin((x * 0.043f) - t * 1.1f + i * 1.7f) * 0.8f;
						clippedPixel(boxX, boxY, boxW, boxH, x, (int)std::lround(y));
					}
				}
			} break;
			case 1: { // Stretch -> expanding ring spacing
				float u = norm01Bipolar(value);
				float expo = 0.55f + u * 1.55f;
				float pulse = 0.25f * std::sin(t * 1.6f);
				for(int i = 1; i <= 6; ++i) {
					float f = (float)i / 6.f;
					float r = 2.f + std::pow(f, expo) * (maxR + pulse * 2.f);
					clippedCircle(boxX, boxY, boxW, boxH, cx, cy, r, 0, 0.78f);
				}
			} break;
			case 2: { // Warp -> directional bend bias
				float v = std::max(-1.f, std::min(1.f, value));
				int lines = 8;
				for(int i = 0; i < lines; ++i) {
					float bx = ix0 + ((i + 0.5f) * iw) / (float)lines;
					float px = bx;
					float py = (float)iy0;
					for(int y = iy0; y < iy0 + ih; ++y) {
						float yy = (float)y - cy;
						float xx = bx + v * yy * 0.22f
							+ std::sin(y * 0.22f + t * 2.1f + i * 0.8f) * (1.0f + std::fabs(v) * 2.8f);
						clippedLine(boxX, boxY, boxW, boxH, px, py, xx, (float)y, 0);
						px = xx; py = (float)y;
					}
				}
			} break;
			case 3: { // Dispersion -> dual-wave desync / moire
				float u = norm01(value, 0.f, 1.f);
				for(int band = 0; band < 5; ++band) {
					float yOff = (band - 2) * 4.2f;
					for(int x = ix0; x < ix0 + iw; ++x) {
						float xf = (float)(x - ix0);
						float y1 = cy + yOff + std::sin(xf * 0.18f + t * 2.0f + band * 0.7f) * 3.0f;
						float y2 = cy + yOff + std::sin(xf * (0.18f + u * 0.11f) - t * (1.6f + u * 1.3f) + band * 1.15f) * (2.0f + u * 2.8f);
						clippedPixel(boxX, boxY, boxW, boxH, x, (int)std::lround(y1));
						clippedPixel(boxX, boxY, boxW, boxH, x, (int)std::lround(y2));
					}
				}
			} break;
			case 4: { // Brightness -> rotating wobbly-to-spiky circle
				float v = std::max(-1.f, std::min(1.f, value));
				float wob = std::max(0.f, -v);
				float spike = std::max(0.f, v);
				float baseR = maxR * 0.68f;
				float rot = t * 0.28f;
				const int segs = 120;
				float px = cx + baseR;
				float py = cy;
				bool havePrev = false;
				for(int i = 0; i <= segs; ++i) {
					float a = rot + (2.f * (float)M_PI * i) / (float)segs;
					float wobble = wob * (std::sin(a * 3.f + t * 1.2f) * 2.4f
						+ std::sin(a * 7.f - t * 1.7f) * 1.2f);
					float s = std::max(0.f, std::sin(a * 12.f));
					float spikes = spike * (1.8f + 3.8f * s * s);
					float r = baseR + wobble + spikes;
					float x = cx + std::cos(a) * r;
					float y = cy + std::sin(a) * r * 0.82f;
					if(havePrev)
						clippedLine(boxX, boxY, boxW, boxH, px, py, x, y, 0);
					px = x;
					py = y;
					havePrev = true;
				}
			} break;
			case 5: { // Partials -> ascending bars with water-wave motion
				int count = 1 + (int)std::lround(norm01(value, 0.f, 32.f) * 15.f);
				float baseY = iy0 + ih - 4.f;
				for(int i = 0; i < count; ++i) {
					float f = (count <= 1) ? 0.5f : (float)i / (float)(count - 1);
					float x = ix0 + 5.f + f * (iw - 10.f);
					float ramp = std::pow(f, 0.82f);
					float wave = 0.5f + 0.5f * std::sin(t * 2.0f + f * 5.8f);
					float h = 5.0f + ramp * (ih - 14.f) + wave * 7.0f;
					float yTop = baseY - h;
					clippedLine(boxX, boxY, boxW, boxH, x, baseY, x, yTop, 0);
				}
			} break;
		}
		return;
	}

	if(page == kPageBodyA2 || page == kPageBodyB2) {
		switch(selected) {
			case 0: // Odd skew -> alternating ring radii
			case 1: { // Even skew -> alternating ring radii, opposite family
				float v = std::max(-1.f, std::min(1.f, value));
				float mag = std::fabs(v);
				float dir = (v >= 0.f) ? 1.f : -1.f;
				bool oddSet = (selected == 0);
				const int rings = 7;
				float innerR = maxR * 0.18f;
				float span = maxR * 0.72f;
				for(int i = 0; i < rings; ++i) {
					float f = (float)i / (float)std::max(1, rings - 1);
					bool family = (((i & 1) == 0) == oddSet);
					float baseR = innerR + f * span;
					float shaped = 0.45f + 0.75f * f;
					float wobble = std::sin(t * 0.85f + i * 0.9f) * (family ? 0.15f + mag * 0.55f : 0.08f);
					float offset = family ? (dir * mag * (1.0f + shaped * 5.2f)) : 0.f;
					float r = std::max(2.5f, baseR + offset + wobble);
					clippedCircle(boxX, boxY, boxW, boxH, cx, cy, r, family && mag > 0.68f ? 1 : 0, 0.78f);

					if(family) {
						for(int k = 0; k < 3; ++k) {
							float a = t * 0.55f + i * 0.5f + k * 2.0943951f;
							int hx = (int)std::lround(cx + std::cos(a) * r);
							int hy = (int)std::lround(cy + std::sin(a) * r * 0.78f);
							clippedSoftBlob(boxX, boxY, boxW, boxH, hx, hy, 0.22f + 0.58f * mag, i * 7 + k);
						}
					}
				}

				// steady center reference ring so the alternating-family displacement reads clearly
				clippedCircle(boxX, boxY, boxW, boxH, cx, cy, innerR - 1.2f, 0, 0.78f);
			} break;
			case 2: { // Collision -> particle clustering / bunching
				float u = norm01(value, 0.f, 1.f);
				const int n = 16;
				for(int i = 0; i < n; ++i) {
					float gx = ix0 + 5.f + (float)(i % 4) * (iw - 10.f) / 3.f;
					float gy = iy0 + 4.f + (float)(i / 4) * (ih - 8.f) / 3.f;
					float ca = t * 0.6f + i * 1.7f;
					float tx = cx + std::cos(ca) * (4.f + (i % 3) * 6.f);
					float ty = cy + std::sin(ca * 1.2f) * (3.f + (i % 2) * 5.f);
					int x = (int)std::lround(gx + (tx - gx) * u);
					int y = (int)std::lround(gy + (ty - gy) * u);
					clippedSoftBlob(boxX, boxY, boxW, boxH, x, y, 0.12f + 0.88f * u, i);
				}
			} break;
			case 3: { // Release amount -> ghost trails
				float u = norm01(value, 0.f, 1.f);
				int trails = 1 + (int)std::lround(u * 3.f);
				for(int k = trails; k >= 0; --k) {
					float tt = t - k * (0.10f + u * 0.18f);
					for(int x = ix0; x < ix0 + iw; ++x) {
						float xf = (float)(x - ix0);
						float y = cy + std::sin(xf * 0.16f + tt * 2.2f) * (2.0f + u * 5.0f);
						clippedPixel(boxX, boxY, boxW, boxH, x, (int)std::lround(y));
					}
				}
			} break;
			case 4: { // Odd/even bias -> interwoven dual-layer braid
				float v = std::max(-1.f, std::min(1.f, value));
				float centerBlend = 1.f - std::fabs(v);
				float oddBoost = std::max(0.f, v);
				float evenBoost = std::max(0.f, -v);
				float oddAmt = centerBlend * 0.30f + oddBoost * 0.95f;
				float evenAmt = centerBlend * 0.30f + evenBoost * 0.95f;
				float prevX = (float)ix0;
				float prevY1 = cy;
				float prevY2 = cy;
				for(int x = ix0; x < ix0 + iw; ++x) {
					float xf = (float)(x - ix0);
					float a = xf * 0.18f + t * 2.0f;
					float y1 = cy + std::sin(a) * 5.5f;
					float y2 = cy + std::sin(a + (float)M_PI) * 5.5f;
					clippedLine(boxX, boxY, boxW, boxH, prevX, prevY1, (float)x, y1, oddBoost > 0.58f ? 1 : 0);
					clippedLine(boxX, boxY, boxW, boxH, prevX, prevY2, (float)x, y2, evenBoost > 0.58f ? 1 : 0);
					if(oddAmt > 0.01f)
						clippedSoftBlob(boxX, boxY, boxW, boxH, x, (int)std::lround(y1), oddAmt, x + 3);
					if(evenAmt > 0.01f)
						clippedSoftBlob(boxX, boxY, boxW, boxH, x, (int)std::lround(y2), evenAmt, x + 11);
					prevX = (float)x;
					prevY1 = y1;
					prevY2 = y2;
				}
			} break;
		}
	}
}


static inline float damperModelShape(float shape, float xn, float m)
{
	shape = std::max(0.f, std::min(1.f, shape));
	xn = std::max(-1.f, std::min(1.f, xn));
	m = std::max(0.f, std::min(1.f, m));
	float rounded = std::pow(shape, 0.62f);
	float pointed = std::pow(shape, 1.55f);
	float shoulder = std::min(1.f, std::pow(shape, 0.92f) * (1.00f + 0.22f * (1.f - xn * xn)));
	float plateau = std::min(1.f, std::pow(shape, 0.78f) + 0.05f * (1.f - xn * xn));
	float p = m * 3.0f;
	int seg = std::max(0, std::min(2, (int)std::floor(p)));
	float frac = p - (float)seg;
	frac = frac * frac * (3.f - 2.f * frac);
	float a = rounded, b = pointed;
	if(seg == 0) { a = rounded; b = pointed; }
	else if(seg == 1) { a = pointed; b = shoulder; }
	else { a = shoulder; b = plateau; }
	return std::max(0.f, std::min(1.f, a + (b - a) * frac));
}

static void drawDamperVisual(int selected, float value, int boxX, int boxY, int boxW, int boxH)
{
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float baseY = iy0 + ih - 5.f;
	const float bellH = ih * 0.60f;
	const float leftX = ix0 + 4.f;
	const float rightX = ix0 + iw - 5.f;
	const float usableW = std::max(8.f, rightX - leftX);

	float center = 0.50f;
	float width = 0.26f;
	float leftPow = 1.45f;
	float rightPow = 1.45f;
	float model = 0.18f;

	if(selected == 0) center = 0.16f + 0.68f * normDamperFreq(value);
	else if(selected == 1) width = 0.12f + 0.34f * normDamperPositive(value, 1.f);
	else if(selected == 2) leftPow = 0.55f + 3.00f * normDamperPositive(value, 4.f);
	else if(selected == 3) rightPow = 0.55f + 3.00f * normDamperPositive(value, 4.f);
	else if(selected == 4) model = normDamperModel(value);

	clippedLine(boxX, boxY, boxW, boxH, leftX, baseY, rightX, baseY, 0);

	for(int pass = 0; pass < 2; ++pass) {
		float contour = 1.f - pass * 0.20f;
		float prevX = 0.f, prevY = 0.f;
		bool havePrev = false;
		for(int x = (int)leftX; x <= (int)rightX; ++x) {
			float xn = ((float)x - leftX) / usableW;
			float dx = (xn - center) / std::max(0.08f, width);
			float adx = std::fabs(dx);
			float sidePow = dx < 0.f ? leftPow : rightPow;
			float raw = 0.f;
			if(adx < 1.f)
				raw = 1.f - std::pow(adx, sidePow);
			float sh = damperModelShape(raw, dx, model) * contour;
			float y = baseY - sh * bellH;
			if(y > baseY)
				y = baseY;
			if(havePrev)
				clippedLine(boxX, boxY, boxW, boxH, prevX, prevY, (float)x, y, 0);
			prevX = (float)x;
			prevY = y;
			havePrev = true;
		}
	}

	if(selected == 0) {
		float peakX = leftX + center * usableW;
		clippedLine(boxX, boxY, boxW, boxH, peakX, baseY - bellH - 1.f, peakX, baseY + 1.f, 0);
	}
}



static inline void drawEnvCurveSegment(int boxX, int boxY, int boxW, int boxH,
	float x0, float y0, float x1, float y1, float bend, int thick = 0)
{
	const int segs = 48;
	float px = x0;
	float py = y0;
	for(int i = 1; i <= segs; ++i) {
		float u = (float)i / (float)segs;
		float ub = (bend >= 0.f)
			? std::pow(u, 1.f + bend * 2.2f)
			: 1.f - std::pow(1.f - u, 1.f + (-bend) * 2.2f);
		float x = x0 + (x1 - x0) * u;
		float y = y0 + (y1 - y0) * ub;
		clippedLine(boxX, boxY, boxW, boxH, px, py, x, y, thick);
		px = x;
		py = y;
	}
}


static inline float envStageNorm(float v)
{
	if(v <= 1.f)
		return std::pow(std::max(0.f, std::min(1.f, v)), 0.60f);
	if(v <= 10.f)
		return std::pow(std::max(0.f, std::min(1.f, v / 5.f)), 0.65f);
	return std::max(0.f, std::min(1.f, std::log1pf(std::max(0.f, v)) / std::log1pf(2000.f)));
}

static inline float curveBendFromStored(float v)
{
	if(v >= -1.f && v <= 1.f)
		return v;
	float u = std::max(0.f, std::min(1.f, v));
	return (u - 0.5f) * 2.f;
}

static void drawSharedAdsrVisual(int activePage, int selected, int boxX, int boxY, int boxW, int boxH)
{
	(const void)activePage;
	(const void)selected;
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float left = ix0 + 4.f;
	const float right = ix0 + iw - 5.f;
	const float top = iy0 + 4.f;
	const float bot = iy0 + ih - 4.f;

	float atk = envStageNorm(gExciterACache[1]);
	float dec = envStageNorm(gExciterACache[2]);
	float sus = std::max(0.f, std::min(1.f, gExciterACache[3]));
	float rel = envStageNorm(gExciterACache[4]);
	float atkCurve = std::max(-1.f, std::min(1.f, curveBendFromStored(gExciterBCache[1])));
	float decCurve = std::max(-1.f, std::min(1.f, curveBendFromStored(gExciterBCache[2])));
	float relCurve = std::max(-1.f, std::min(1.f, curveBendFromStored(gExciterBCache[3])));

	const float totalW = std::max(36.f, right - left);
	const float minSegW = 7.f;
	const float minSusW = 14.f;
	const float variableW = std::max(6.f, totalW - (minSegW * 3.f) - minSusW);
	const float sumAdr = std::max(0.001f, atk + dec + rel);
	float atkW = minSegW + variableW * (atk / sumAdr);
	float decW = minSegW + variableW * (dec / sumAdr);
	float relW = minSegW + variableW * (rel / sumAdr);
	float susW = totalW - (atkW + decW + relW);
	if(susW < minSusW) {
		float over = minSusW - susW;
		float reducible = std::max(0.001f, (atkW - minSegW) + (decW - minSegW) + (relW - minSegW));
		atkW -= over * std::max(0.f, atkW - minSegW) / reducible;
		decW -= over * std::max(0.f, decW - minSegW) / reducible;
		relW -= over * std::max(0.f, relW - minSegW) / reducible;
		susW = minSusW;
	}

	const float x0 = left;
	const float x1 = x0 + atkW;
	const float x2 = x1 + decW;
	const float x3 = x2 + susW;
	const float x4 = std::min(right, x3 + relW);

	const float yBase = bot;
	const float yPeak = top;
	float ySus = bot - 2.f - sus * (ih * 0.58f);
	if(ySus < yPeak + 4.f)
		ySus = yPeak + 4.f;

	drawEnvCurveSegment(boxX, boxY, boxW, boxH, x0, yBase, x1, yPeak, atkCurve, 0);
	drawEnvCurveSegment(boxX, boxY, boxW, boxH, x1, yPeak, x2, ySus, decCurve, 0);
	clippedLine(boxX, boxY, boxW, boxH, x2, ySus, x3, ySus, 0);
	drawEnvCurveSegment(boxX, boxY, boxW, boxH, x3, ySus, x4, yBase, relCurve, 0);

	clippedHollowNode(boxX, boxY, boxW, boxH, (int)std::lround(x1), (int)std::lround(yPeak));
	clippedHollowNode(boxX, boxY, boxW, boxH, (int)std::lround(x2), (int)std::lround(ySus));
	clippedHollowNode(boxX, boxY, boxW, boxH, (int)std::lround(x3), (int)std::lround(ySus));
}

static void drawBlendStarNoiseVisual(float value, int boxX, int boxY, int boxW, int boxH)
{
	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	float v = std::max(-1.f, std::min(1.f, value));
	float impulse = std::max(0.f, -v);
	float noise = std::max(0.f, v);
	float mix = 1.f - std::fabs(v);
	const int count = 28;
	for(int i = 0; i < count; ++i) {
		float fi = (float)i / (float)count;
		float a = fi * 6.2831853f;
		float nrx = std::sin(a * 3.1f + t * (4.2f + fi * 3.0f) + i * 0.37f);
		float nry = std::cos(a * 2.7f - t * (3.5f + fi * 2.1f) + i * 0.51f);
		float noiseR = 3.0f + noise * (6.0f + 8.0f * ((i % 5) / 4.f)) + mix * 3.0f;
		float nx = cx + nrx * noiseR + std::sin(t * 7.5f + i * 1.9f) * noise * 2.0f;
		float ny = cy + nry * noiseR * 0.78f + std::cos(t * 6.3f + i * 1.3f) * noise * 1.6f;

		int arm = i & 3;
		float along = ((i / 4) + 1.f) / 8.f;
		float sx = cx, sy = cy;
		float armLen = 3.5f + along * std::min(iw, ih) * 0.28f;
		if(arm == 0) { sx = cx; sy = cy - armLen; }
		else if(arm == 1) { sx = cx + armLen; sy = cy; }
		else if(arm == 2) { sx = cx; sy = cy + armLen; }
		else { sx = cx - armLen; sy = cy; }
		// little width so the star has body, not a single-pixel skeleton
		if((i & 1) == 0) {
			if(arm == 0 || arm == 2) sx += ((i & 2) ? 1.f : -1.f);
			else sy += ((i & 2) ? 1.f : -1.f);
		}

		float morph = impulse + mix * 0.35f;
		float x = nx + (sx - nx) * morph;
		float y = ny + (sy - ny) * morph;
		clippedPixel(boxX, boxY, boxW, boxH, (int)std::lround(x), (int)std::lround(y));
		if(impulse > 0.62f && (i % 3) == 0)
			clippedPixel(boxX, boxY, boxW, boxH, (int)std::lround((x + cx) * 0.5f), (int)std::lround((y + cy) * 0.5f));
	}
}

static void drawImpulseShapeVisual(float value, int boxX, int boxY, int boxW, int boxH)
{
	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	float u = std::max(0.f, std::min(1.f, value));
	float tall = 4.0f + (1.f - u) * 8.0f;
	float wide = 3.0f + u * 11.0f;
	float waist = 1.0f + u * 3.0f;
	// centered four-sided spark/diamond pulse
	clippedLine(boxX, boxY, boxW, boxH, cx, cy - tall, cx + waist, cy, 0);
	clippedLine(boxX, boxY, boxW, boxH, cx + waist, cy, cx, cy + tall, 0);
	clippedLine(boxX, boxY, boxW, boxH, cx, cy + tall, cx - waist, cy, 0);
	clippedLine(boxX, boxY, boxW, boxH, cx - waist, cy, cx, cy - tall, 0);
	clippedLine(boxX, boxY, boxW, boxH, cx - wide, cy, cx - waist, cy, u > 0.55f ? 1 : 0);
	clippedLine(boxX, boxY, boxW, boxH, cx + waist, cy, cx + wide, cy, u > 0.55f ? 1 : 0);
	if(u > 0.38f) {
		float wing = 2.0f + u * 4.0f;
		clippedLine(boxX, boxY, boxW, boxH, cx - wide * 0.55f, cy - wing, cx, cy, 0);
		clippedLine(boxX, boxY, boxW, boxH, cx - wide * 0.55f, cy + wing, cx, cy, 0);
		clippedLine(boxX, boxY, boxW, boxH, cx + wide * 0.55f, cy - wing, cx, cy, 0);
		clippedLine(boxX, boxY, boxW, boxH, cx + wide * 0.55f, cy + wing, cx, cy, 0);
	}
	if(u > 0.70f) {
		for(int k = 0; k < 4; ++k) {
			float a = t * 1.4f + k * 1.5707963f;
			int px = (int)std::lround(cx + std::cos(a) * (1.5f + u * 2.0f));
			int py = (int)std::lround(cy + std::sin(a) * (1.2f + u * 1.7f));
			clippedPixel(boxX, boxY, boxW, boxH, px, py);
		}
	}
}

static void drawExciterVisual(int page, int selected, float value, int boxX, int boxY, int boxW, int boxH)
{
	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float left = ix0 + 4.f;
	const float right = ix0 + iw - 5.f;
	const float top = iy0 + 3.f;
	const float bot = iy0 + ih - 4.f;
	const float cy = iy0 + ih * 0.5f;

	if(page == kPageExciterA) {
		switch(selected) {
			case 0:
				drawBlendStarNoiseVisual(value, boxX, boxY, boxW, boxH);
				break;
			case 1:
			case 2:
			case 3:
			case 4:
				drawSharedAdsrVisual(page, selected, boxX, boxY, boxW, boxH);
				break;
			case 5: { // Noise color
				float u = norm01(value, 0.f, 1.f);
				for(int x = (int)left; x <= (int)right; ++x) {
					float xf = (float)(x - left);
					float low = std::sin(xf * (0.06f + u * 0.03f) + t * (1.3f + u * 0.8f)) * (5.2f - u * 2.0f);
					float hi = std::sin(xf * (0.35f + u * 0.55f) - t * (2.0f + u * 5.0f)) * (0.8f + u * 3.2f);
					float y = cy + low + hi;
					clippedPixel(boxX, boxY, boxW, boxH, x, (int)std::lround(y));
				}
			} break;
		}
		return;
	}

	if(page == kPageExciterB) {
		switch(selected) {
			case 0:
				drawImpulseShapeVisual(value, boxX, boxY, boxW, boxH);
				break;
			case 1:
			case 2:
			case 3:
				drawSharedAdsrVisual(page, selected, boxX, boxY, boxW, boxH);
				break;
			default: {
				clippedLine(boxX, boxY, boxW, boxH, left + 10.f, cy, right - 10.f, cy, 0);
				clippedLine(boxX, boxY, boxW, boxH, (left + right) * 0.5f, top + 8.f, (left + right) * 0.5f, bot - 8.f, 0);
			} break;
		}
	}
}

static inline float smoothVisualValue(float& state, float target)
{
	if(!std::isfinite(state))
		state = target;
	float diff = std::fabs(target - state);
	float alpha = 0.16f + std::min(0.34f, diff * 0.42f);
	state += (target - state) * alpha;
	return state;
}

static void drawSpaceRoomFrame(int boxX, int boxY, int boxW, int boxH,
	float depth01, float wobble, float t,
	float& frontL, float& frontT, float& frontR, float& frontB,
	float& backL, float& backT, float& backR, float& backB)
{
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;

	// The overlay box border itself is the room's front wall.
	frontL = (float)boxX;
	frontR = (float)(boxX + boxW - 1);
	frontT = (float)boxY;
	frontB = (float)(boxY + boxH - 1);

	depth01 = std::max(0.f, std::min(1.f, depth01));
	float insetX = 3.f + depth01 * 19.f;
	float insetY = 2.f + depth01 * 9.f;
	float driftX = std::sin(t * 0.85f + depth01 * 2.7f) * wobble;
	float driftY = std::cos(t * 0.70f + depth01 * 3.1f) * wobble * 0.6f;

	backL = ix0 + insetX + driftX;
	backR = ix0 + iw - 1.f - insetX + driftX;
	backT = iy0 + insetY + driftY;
	backB = iy0 + ih - 1.f - insetY + driftY;

	if(backR < backL + 8.f) {
		float c = (backL + backR) * 0.5f;
		backL = c - 4.f;
		backR = c + 4.f;
	}
	if(backB < backT + 8.f) {
		float c = (backT + backB) * 0.5f;
		backT = c - 4.f;
		backB = c + 4.f;
	}

	// Only draw the back wall and depth connectors. The border rectangle already draws the front wall.
	clippedLine(boxX, boxY, boxW, boxH, backL, backT, backR, backT, 0);
	clippedLine(boxX, boxY, boxW, boxH, backR, backT, backR, backB, 0);
	clippedLine(boxX, boxY, boxW, boxH, backR, backB, backL, backB, 0);
	clippedLine(boxX, boxY, boxW, boxH, backL, backB, backL, backT, 0);

	clippedLine(boxX, boxY, boxW, boxH, ix0 + 1.f, iy0 + 1.f, backL, backT, 0);
	clippedLine(boxX, boxY, boxW, boxH, ix0 + iw - 2.f, iy0 + 1.f, backR, backT, 0);
	clippedLine(boxX, boxY, boxW, boxH, ix0 + 1.f, iy0 + ih - 2.f, backL, backB, 0);
	clippedLine(boxX, boxY, boxW, boxH, ix0 + iw - 2.f, iy0 + ih - 2.f, backR, backB, 0);
}

static void drawSpaceSparkle(int boxX, int boxY, int boxW, int boxH, int x, int y, int seed = 0)
{
	knockedOutPixel(boxX, boxY, boxW, boxH, x, y, 1);
	knockedOutPixel(boxX, boxY, boxW, boxH, x - 1, y, 1);
	knockedOutPixel(boxX, boxY, boxW, boxH, x + 1, y, 1);
	knockedOutPixel(boxX, boxY, boxW, boxH, x, y - 1, 1);
	knockedOutPixel(boxX, boxY, boxW, boxH, x, y + 1, 1);
	if((seed & 1) == 0) {
		knockedOutPixel(boxX, boxY, boxW, boxH, x - 1, y - 1, 1);
		knockedOutPixel(boxX, boxY, boxW, boxH, x + 1, y + 1, 1);
	}
	if((seed & 2) == 0) {
		knockedOutPixel(boxX, boxY, boxW, boxH, x + 1, y - 1, 1);
		knockedOutPixel(boxX, boxY, boxW, boxH, x - 1, y + 1, 1);
	}
	// Hollow center helps the sparkle stay readable against room lines.
	clippedPixelSet(boxX, boxY, boxW, boxH, x, y, false);
}

static void drawSpaceVisual(int selected, float value, int boxX, int boxY, int boxW, int boxH)
{
	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;
	float u = (selected == 5) ? norm01(value, 0.f, 1.f) : norm01(value, 0.f, 1.f);
	u = smoothVisualValue(gSpaceSmooth[std::max(0, std::min(selected, kParamCount - 1))], u);

	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	const float frontInnerL = (float)ix0 + 1.f;
	const float frontInnerT = (float)iy0 + 1.f;
	const float frontInnerR = (float)ix0 + iw - 2.f;
	const float frontInnerB = (float)iy0 + ih - 2.f;

	float frontL, frontT, frontR, frontB, backL, backT, backR, backB;
	float depth = 0.18f + 0.52f * u;
	if(selected == 0)
		depth = 0.06f + 0.82f * u;
	float wobble = (selected == 0) ? 0.f : (selected == 2 ? 0.12f + 0.22f * u : 0.08f);
	drawSpaceRoomFrame(boxX, boxY, boxW, boxH, depth, wobble, t,
		frontL, frontT, frontR, frontB, backL, backT, backR, backB);

	auto eraseLinePattern = [&](float x0, float y0, float x1, float y1, int period, int eraseCount, int phase)
	{
		if(period <= 1 || eraseCount <= 0)
			return;
		int steps = (int)std::max(std::fabs(x1 - x0), std::fabs(y1 - y0));
		if(steps < 1) steps = 1;
		for(int i = 0; i <= steps; ++i) {
			if(((i + phase) % period) < eraseCount) {
				float tt = (float)i / (float)steps;
				int x = (int)std::lround(x0 + (x1 - x0) * tt);
				int y = (int)std::lround(y0 + (y1 - y0) * tt);
				clippedPixelSet(boxX, boxY, boxW, boxH, x, y, false);
			}
		}
	};

	auto drawPatternLine = [&](float x0, float y0, float x1, float y1, int period, int onCount, int phase, int halo)
	{
		if(period <= 1 || onCount >= period) {
			knockedOutLine(boxX, boxY, boxW, boxH, x0, y0, x1, y1, 0, halo);
			return;
		}
		int steps = (int)std::max(std::fabs(x1 - x0), std::fabs(y1 - y0));
		if(steps < 1) steps = 1;
		for(int i = 0; i <= steps; ++i) {
			if(((i + phase) % period) < onCount) {
				float tt = (float)i / (float)steps;
				int x = (int)std::lround(x0 + (x1 - x0) * tt);
				int y = (int)std::lround(y0 + (y1 - y0) * tt);
				knockedOutPixel(boxX, boxY, boxW, boxH, x, y, halo);
			}
		}
	};

	auto drawPatternCircle = [&](float rc, int period, int onCount, int phase, float yScale, int halo)
	{
		if(rc <= 0.5f)
			return;
		const int n = 96;
		for(int i = 0; i < n; ++i) {
			float a0 = (2.f * (float)M_PI * i) / (float)n;
			float a1 = (2.f * (float)M_PI * (i + 1)) / (float)n;
			float x0 = cx + std::cos(a0) * rc;
			float y0 = cy + std::sin(a0) * rc * yScale;
			float x1 = cx + std::cos(a1) * rc;
			float y1 = cy + std::sin(a1) * rc * yScale;
			drawPatternLine(x0, y0, x1, y1, period, onCount, phase + i, halo);
		}
	};

	auto drawPlane = [&](float planeDepth, int period, int onCount, int phase, int halo)
	{
		planeDepth = std::max(0.f, std::min(1.f, planeDepth));
		float l = frontInnerL + (backL - frontInnerL) * planeDepth;
		float r = frontInnerR + (backR - frontInnerR) * planeDepth;
		float tp = frontInnerT + (backT - frontInnerT) * planeDepth;
		float b = frontInnerB + (backB - frontInnerB) * planeDepth;
		drawPatternLine(l, tp, r, tp, period, onCount, phase + 0, halo);
		drawPatternLine(r, tp, r, b, period, onCount, phase + 1, halo);
		drawPatternLine(r, b, l, b, period, onCount, phase + 2, halo);
		drawPatternLine(l, b, l, tp, period, onCount, phase + 3, halo);
	};

	switch(selected) {
		case 0: {
			// Space Size: nothing extra. The single 1-pixel room frame is the visual.
		} break;

		case 1: {
			// Space Decay: concentric halos with cleaner spacing and lighter outer dithering.
			int rings = 2 + (int)std::lround(u * 4.f);
			float baseR = 4.5f + 0.35f * std::sin(t * 1.15f);
			float gap = 1.4f + 0.55f * u;
			for(int i = 0; i < rings; ++i) {
				float rr = baseR + i * gap;
				int period = 1;
				int onCount = 1;
				if(i >= 1) {
					period = 4 + std::min(i, 2);
					onCount = std::max(1, period - (1 + i / 2));
				}
				drawPatternCircle(rr, period, onCount, i * 3 + (int)std::lround(t * 6.f), 0.68f, 1);
			}
		} break;

		case 2: {
			// Diffusion stays as floating sparkles, but separated from room lines.
			float roomW = std::max(8.f, backR - backL);
			float roomH = std::max(8.f, backB - backT);
			int count = 7 + (int)std::lround(u * 20.f);
			for(int i = 0; i < count; ++i) {
				float fi = (float)i / (float)std::max(1, count - 1);
				float px = backL + 2.f + std::fmod(fi * 47.3f + std::sin(t * 0.7f + i * 0.91f) * (2.0f + u * 4.5f) + i * 9.1f, std::max(4.f, roomW - 4.f));
				float py = backT + 2.f + std::fmod(fi * 29.7f + std::cos(t * 0.9f + i * 1.23f) * (1.5f + u * 3.6f) + i * 5.7f, std::max(4.f, roomH - 4.f));
				if((i % 3) == 0 && u > 0.25f)
					drawSpaceSparkle(boxX, boxY, boxW, boxH, (int)std::lround(px), (int)std::lround(py), i);
				else
					knockedOutSoftBlob(boxX, boxY, boxW, boxH, (int)std::lround(px), (int)std::lround(py), 0.10f + u * 0.56f, i, 1);
			}
		} break;

		case 3: {
			// Damping: fade/break the perspective lines themselves with fake dithering.
			int period = 6;
			int eraseCount = (int)std::lround(u * 4.f);
			eraseLinePattern(backL, backT, backR, backT, period, eraseCount, 0);
			eraseLinePattern(backR, backT, backR, backB, period, eraseCount, 1);
			eraseLinePattern(backR, backB, backL, backB, period, eraseCount, 2);
			eraseLinePattern(backL, backB, backL, backT, period, eraseCount, 3);
			eraseLinePattern(frontInnerL, frontInnerT, backL, backT, period, eraseCount, 0);
			eraseLinePattern(frontInnerR, frontInnerT, backR, backT, period, eraseCount, 1);
			eraseLinePattern(frontInnerL, frontInnerB, backL, backB, period, eraseCount, 2);
			eraseLinePattern(frontInnerR, frontInnerB, backR, backB, period, eraseCount, 3);

			// A light secondary dither pass helps the fade feel smoother instead of just broken.
			if(u > 0.18f) {
				int phase = (int)std::lround(t * 7.f) & 3;
				drawPlane(0.48f, 6, std::max(1, 4 - eraseCount), phase, 1);
			}
		} break;

		case 4: {
			// Onset: a wavefront plane travels from the front wall toward the back wall.
			float leadDepth = 0.08f + (1.f - u) * 0.70f;
			drawPlane(leadDepth, 1, 1, 0, 1);
			if(u < 0.96f)
				drawPlane(std::min(1.f, leadDepth + 0.08f), 5, 3, 1 + (int)std::lround(t * 5.f), 1);
			if(u < 0.78f)
				drawPlane(std::min(1.f, leadDepth + 0.16f), 6, 2, 3 + (int)std::lround(t * 5.f), 1);
		} break;

		case 5: {
			// Wet/Dry: dry = one direct front plane, wet = multiple deeper room planes.
			float dry = 1.f - u;
			float wet = u;
			if(dry > 0.04f) {
				float dryDepth = 0.05f + dry * 0.08f;
				drawPlane(dryDepth, 1, 1, 0, 1);
			}
			if(wet > 0.04f) {
				int planes = 1 + (int)std::lround(wet * 3.f);
				for(int i = 0; i < planes; ++i) {
					float fi = (planes <= 1) ? 0.f : (float)i / (float)(planes - 1);
					float d = 0.20f + fi * (0.42f + wet * 0.08f);
					int period = (i == 0) ? 4 : 5 + i;
					int onCount = std::max(1, period - 2 - i / 2);
					drawPlane(d, period, onCount, i * 2 + (int)std::lround(t * 4.f), 1);
				}
			}
		} break;
	}
}



static inline float satClamp01(float v)
{
	return std::max(0.f, std::min(1.f, v));
}

static inline float satClampBi(float v)
{
	return std::max(-1.f, std::min(1.f, v));
}

static inline float satSmootherStep(float t)
{
	t = satClamp01(t);
	return t * t * (3.f - 2.f * t);
}

static inline float satBaseWave(float phase, float bright)
{
	bright = satClamp01(bright);
	float y = std::sin(phase);
	y += (0.08f + bright * 0.06f) * std::sin(phase * 3.f);
	y += bright * 0.04f * std::sin(phase * 5.f);
	return y * 0.86f;
}

static inline float satTriangleWave(float phase)
{
	return (2.f / (float)M_PI) * std::asin(std::sin(phase));
}


static inline float satToneWave(float phase, float tone)
{
	tone = satClampBi(tone);
	float dark = std::max(0.f, -tone);
	float bright = std::max(0.f, tone);

	// Keep Tone continuous and distinct from Curve:
	// dark side = rounder / chunkier harmonic body
	// bright side = crisper pointed continuous wave, but no saw discontinuity
	float darkWave = std::sin(phase) * (1.f - dark * 0.14f)
		+ (0.18f + dark * 0.10f) * std::sin(phase * 2.f)
		+ 0.05f * (1.f + dark * 0.4f) * std::sin(phase * 3.f);
	darkWave = std::tanh(darkWave * (0.92f + dark * 0.20f));

	float brightWave = satTriangleWave(phase) * (0.82f + bright * 0.06f)
		+ 0.16f * bright * std::sin(phase * 3.f)
		+ 0.07f * bright * std::sin(phase * 5.f);
	brightWave = satClampBi(brightWave);

	float mix = satSmootherStep(bright);
	return satClampBi(darkWave * (1.f - mix) + brightWave * mix);
}

static inline float satShapeSample(float x, float drive, float curve, float asym)
{
	drive = satClamp01(drive);
	curve = satClamp01(curve);
	asym = satClampBi(asym);
	float posAmt = std::max(0.f, asym);
	float negAmt = std::max(0.f, -asym);
	float posGain = 1.f + posAmt * 1.35f - negAmt * 0.25f;
	float negGain = 1.f + negAmt * 1.35f - posAmt * 0.25f;
	float bias = asym * 0.22f;
	float g = 1.f + drive * 5.8f;
	float xb = x + bias;
	float xin = xb * g * (xb >= 0.f ? posGain : negGain);
	float soft = std::tanh(xin);
	float hard = (xin >= 0.f) ? 1.f : -1.f;
	float mix = satSmootherStep(curve);
	float y = soft * (1.f - mix) + hard * mix;
	y -= asym * 0.10f;
	return std::max(-1.f, std::min(1.f, y));
}

static void satDrawHLine(int boxX, int boxY, int boxW, int boxH, int x0, int x1, int y)
{
	if(x0 > x1)
		std::swap(x0, x1);
	for(int x = x0; x <= x1; ++x)
		clippedPixel(boxX, boxY, boxW, boxH, x, y);
}

static inline int satBayer4x4(int x, int y, int phase = 0)
{
	static const int m[16] = {
		0, 8, 2, 10,
		12, 4, 14, 6,
		3, 11, 1, 9,
		15, 7, 13, 5
	};
	int xx = (x + phase) & 3;
	int yy = (y + ((phase >> 1) & 3)) & 3;
	return m[yy * 4 + xx];
}

static inline bool satKeepDitheredPixel(int x, int y, float opacity, int phase = 0)
{
	opacity = satClamp01(opacity);
	if(opacity <= 0.f)
		return false;
	if(opacity >= 0.999f)
		return true;
	int keep = (int)std::floor(opacity * 16.f + 0.0001f);
	keep = std::max(0, std::min(16, keep));
	return satBayer4x4(x, y, phase) < keep;
}

static inline void satPlotAlphaPixel(int boxX, int boxY, int boxW, int boxH,
	int x, int y, float opacity, int phase = 0)
{
	if(satKeepDitheredPixel(x, y, opacity, phase))
		clippedPixel(boxX, boxY, boxW, boxH, x, y);
}

static inline void satPlotAlphaPixelThick(int boxX, int boxY, int boxW, int boxH,
	int x, int y, float opacity, int thicknessRadius, int phase = 0)
{
	thicknessRadius = std::max(0, thicknessRadius);
	for(int yy = y - thicknessRadius; yy <= y + thicknessRadius; ++yy)
		satPlotAlphaPixel(boxX, boxY, boxW, boxH, x, yy, opacity, phase);
}

static void satDrawWaveTraceEx(int boxX, int boxY, int boxW, int boxH,
	int left, int right, float cy, float ampPx,
	float drive, float curve, float asym, float bright,
	float yOffset = 0.f, float phaseOffset = 0.f,
	int clipTopY = -10000, int clipBotY = 10000,
	float opacity = 1.f, int ditherPhase = 0, int maxJoinDy = 64)
{
	int prevY = (int)std::lround(cy + yOffset);
	bool prevClipped = false;
	bool havePrev = false;
	for(int x = left; x <= right; ++x) {
		float fi = (right == left) ? 0.f : (float)(x - left) / (float)(right - left);
		float phase = fi * 2.f * (float)M_PI + phaseOffset;
		float base = satBaseWave(phase, bright);
		float shaped = satShapeSample(base, drive, curve, asym);
		int rawY = (int)std::lround(cy + yOffset - shaped * ampPx);
		bool clipped = (rawY < clipTopY || rawY > clipBotY);
		int y = std::max(clipTopY, std::min(clipBotY, rawY));
		satPlotAlphaPixel(boxX, boxY, boxW, boxH, x, y, opacity, ditherPhase);
		if(havePrev) {
			int dy = y - prevY;
			bool oppositeRails = (prevY == clipTopY && y == clipBotY) || (prevY == clipBotY && y == clipTopY);
			bool allowJoin = std::abs(dy) <= maxJoinDy && !(oppositeRails && (prevClipped || clipped));
			if(allowJoin) {
				if(y > prevY + 1) {
					for(int yy = prevY + 1; yy < y; ++yy)
						satPlotAlphaPixel(boxX, boxY, boxW, boxH, x - 1, yy, opacity, ditherPhase);
				} else if(y < prevY - 1) {
					for(int yy = prevY - 1; yy > y; --yy)
						satPlotAlphaPixel(boxX, boxY, boxW, boxH, x - 1, yy, opacity, ditherPhase);
				}
			}
		}
		prevY = y;
		prevClipped = clipped;
		havePrev = true;
	}
}

static void satDrawWaveTrace(int boxX, int boxY, int boxW, int boxH,
	int left, int right, float cy, float ampPx,
	float drive, float curve, float asym, float bright,
	float yOffset = 0.f, float phaseOffset = 0.f,
	int clipTopY = -10000, int clipBotY = 10000)
{
	satDrawWaveTraceEx(boxX, boxY, boxW, boxH, left, right, cy, ampPx,
		drive, curve, asym, bright, yOffset, phaseOffset, clipTopY, clipBotY,
		1.f, 0, 64);
}

static void satDrawToneTrace(int boxX, int boxY, int boxW, int boxH,
	int left, int right, float cy, float ampPx, float tone,
	float drive, float curve, float yOffset = 0.f)
{
	int prevY = (int)std::lround(cy + yOffset);
	bool havePrev = false;
	for(int x = left; x <= right; ++x) {
		float fi = (right == left) ? 0.f : (float)(x - left) / (float)(right - left + 1);
		float phase = fi * 2.f * (float)M_PI;
		float base = satToneWave(phase, tone);
		float shaped = satShapeSample(base, drive, curve, 0.f);
		int y = (int)std::lround(cy + yOffset - shaped * ampPx);
		clippedPixel(boxX, boxY, boxW, boxH, x, y);
		if(havePrev) {
			if(y > prevY + 1) {
				for(int yy = prevY + 1; yy < y; ++yy)
					clippedPixel(boxX, boxY, boxW, boxH, x - 1, yy);
			} else if(y < prevY - 1) {
				for(int yy = prevY - 1; yy > y; --yy)
					clippedPixel(boxX, boxY, boxW, boxH, x - 1, yy);
			}
		}
		prevY = y;
		havePrev = true;
	}
}


static void satDrawAsymTrace(int boxX, int boxY, int boxW, int boxH,
	int left, int right, float cy, float ampPx, float asym)
{
	const float drive = 1.0f;
	const float curve = 0.76f;
	const float bright = 0.03f;
	const float asymAmt = satClampBi(asym * 2.35f);
	float shapedMin =  9999.f;
	float shapedMax = -9999.f;
	for(int x = left; x <= right; ++x) {
		float fi = (right == left) ? 0.f : (float)(x - left) / (float)(right - left);
		float phase = fi * 2.f * (float)M_PI;
		float base = satBaseWave(phase, bright);
		float shaped = satShapeSample(base, drive, curve, asymAmt);
		shapedMin = std::min(shapedMin, shaped);
		shapedMax = std::max(shapedMax, shaped);
	}
	float center = 0.5f * (shapedMin + shapedMax);
	int prevY = (int)std::lround(cy);
	bool havePrev = false;
	for(int x = left; x <= right; ++x) {
		float fi = (right == left) ? 0.f : (float)(x - left) / (float)(right - left);
		float phase = fi * 2.f * (float)M_PI;
		float base = satBaseWave(phase, bright);
		float shaped = satShapeSample(base, drive, curve, asymAmt) - center;
		int y = (int)std::lround(cy - shaped * ampPx);
		clippedPixel(boxX, boxY, boxW, boxH, x, y);
		if(havePrev) {
			if(y > prevY + 1) {
				for(int yy = prevY + 1; yy < y; ++yy)
					clippedPixel(boxX, boxY, boxW, boxH, x - 1, yy);
			} else if(y < prevY - 1) {
				for(int yy = prevY - 1; yy > y; --yy)
					clippedPixel(boxX, boxY, boxW, boxH, x - 1, yy);
			}
		}
		prevY = y;
		havePrev = true;
	}
}

static float satPureSine(float phase)
{
	return std::sin(phase);
}

static float satWetSquare(float phase)
{
	return satShapeSample(std::sin(phase), 1.f, 1.f, 0.f);
}

static void satDrawCustomWaveLane(int boxX, int boxY, int boxW, int boxH,
	int left, int right, float cy, float ampPx,
	float (*waveFn)(float), float opacity, int thicknessRadius, int ditherPhase)
{
	int prevY = (int)std::lround(cy);
	bool havePrev = false;
	for(int x = left; x <= right; ++x) {
		float fi = (right == left) ? 0.f : (float)(x - left) / (float)(right - left);
		float phase = fi * 2.f * (float)M_PI;
		float shaped = waveFn(phase);
		int y = (int)std::lround(cy - shaped * ampPx);
		satPlotAlphaPixelThick(boxX, boxY, boxW, boxH, x, y, opacity, thicknessRadius, ditherPhase);
		if(havePrev) {
			if(y > prevY + 1) {
				for(int yy = prevY + 1; yy < y; ++yy)
					satPlotAlphaPixelThick(boxX, boxY, boxW, boxH, x - 1, yy, opacity, thicknessRadius, ditherPhase);
			} else if(y < prevY - 1) {
				for(int yy = prevY - 1; yy > y; --yy)
					satPlotAlphaPixelThick(boxX, boxY, boxW, boxH, x - 1, yy, opacity, thicknessRadius, ditherPhase);
			}
		}
		prevY = y;
		havePrev = true;
	}
}

static void drawSaturationVisual(int selected, float value, int boxX, int boxY, int boxW, int boxH)
{
	float raw = (selected >= 3) ? norm01Bipolar(value) : norm01(value, 0.f, 1.f);
	float u = smoothVisualValue(gSatSmooth[std::max(0, std::min(selected, kParamCount - 1))], raw);

	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const int left = ix0 + 5;
	const int right = ix0 + iw - 6;
	const float cy = iy0 + ih * 0.5f;
	const float amp = std::max(7.f, ih * 0.28f);
	const int railLeft = left + 2;
	const int railRight = right - 2;

	switch(selected) {
		case 0: { // Drive = waveform gets pushed harder and flattens
			float amt = satClamp01(u);
			float drive = 0.08f + amt * 0.98f;
			float ampMul = 0.90f + amt * 0.58f;
			satDrawWaveTrace(boxX, boxY, boxW, boxH, left, right, cy, amp * ampMul,
				drive, 0.22f, 0.f, 0.06f, 0.f, 0.f);
		} break;

		case 1: { // Threshold = clipping rails move inward and squash the waveform
			float amt = satClamp01(u);
			float railPos = (0.78f - amt * 0.56f) * amp;
			int yTop = (int)std::lround(cy - railPos);
			int yBot = (int)std::lround(cy + railPos);
			satDrawHLine(boxX, boxY, boxW, boxH, railLeft, railRight, yTop);
			satDrawHLine(boxX, boxY, boxW, boxH, railLeft, railRight, yBot);
			satDrawWaveTraceEx(boxX, boxY, boxW, boxH, left, right, cy, amp * 1.45f,
				0.98f, 0.72f, 0.f, 0.06f, 0.f, 0.f, yTop, yBot,
				1.f, 0, 3);
		} break;

		case 2: { // Curve = rounded distortion morphs toward literal square wave
			float amt = satClamp01(u);
			satDrawWaveTrace(boxX, boxY, boxW, boxH, left, right, cy, amp * (0.98f + amt * 0.12f),
				0.96f, amt, 0.f, 0.03f, 0.f, 0.f);
		} break;

		case 3: { // Asymmetry = stronger, centered, obvious top/bottom mismatch
			float amt = satClampBi(u);
			satDrawAsymTrace(boxX, boxY, boxW, boxH, left, right, cy, amp * 1.34f, amt);
		} break;

		case 4: { // Tone = waveform family changes, not just clipping hardness
			float amt = satClampBi(u);
			float dark = std::max(0.f, -amt);
			float bright = std::max(0.f, amt);
			float drive = 0.34f + bright * 0.28f;
			float curve = 0.08f + bright * 0.14f;
			satDrawToneTrace(boxX, boxY, boxW, boxH, left, right, cy,
				amp * (0.96f + dark * 0.08f + bright * 0.04f), amt, drive, curve, 0.f);
		} break;

		case 5: { // Wet/Dry = 0 shows both fully; positive fades sine, negative fades square
			float bias = satClampBi(u);
			float laneSep = std::max(6.f, ih * 0.19f);
			float laneAmp = std::max(3.8f, std::min(amp * 0.52f, laneSep - 1.6f));
			float topCy = cy - laneSep;
			float botCy = cy + laneSep;

			// Bipolar rule requested by user:
			// bias = 0   -> both fully visible
			// bias > 0   -> sine fades away, square stays solid
			// bias < 0   -> square fades away, sine stays solid
			float dryOpacity = (bias > 0.f) ? (1.f - bias) : 1.f;
			float wetOpacity = (bias < 0.f) ? (1.f + bias) : 1.f;
			dryOpacity = satClamp01(dryOpacity);
			wetOpacity = satClamp01(wetOpacity);

			if(dryOpacity > 0.01f)
				satDrawCustomWaveLane(boxX, boxY, boxW, boxH, left, right, topCy, laneAmp,
					satPureSine, dryOpacity, 0, 0);
			if(wetOpacity > 0.01f)
				satDrawCustomWaveLane(boxX, boxY, boxW, boxH, left, right, botCy, laneAmp,
					satWetSquare, wetOpacity, 0, 2);
		} break;
	}
}


static inline int lfoPageSlot(int page)
{
	return (page == kPageModLfo2) ? 1 : 0;
}

static inline float lfoNormRate(float v)
{
	if(v <= 1.f)
		return std::max(0.f, std::min(1.f, v));
	return std::max(0.f, std::min(1.f, std::log1pf(std::max(0.f, v)) / std::log1pf(20.f)));
}

static inline int lfoShapeIndex(float v)
{
	int s = (int)std::lround(v);
	if(s < 1) s = 1;
	if(s > 5) s = 5;
	return s - 1;
}

static inline int lfoModeIndex(float v)
{
	return ((int)std::lround(v) <= 0) ? 0 : 1;
}

static inline float lfoStepHash(int n)
{
	float x = std::sin((float)n * 12.9898f + 78.233f) * 43758.5453f;
	return -1.f + 2.f * (x - std::floor(x));
}

static inline float lfoSampleShape(int shape, float ph)
{
	ph = ph - std::floor(ph);
	switch(shape) {
		case 0: return -1.f + 2.f * ph; // saw up
		case 1: return (ph < 0.5f) ? 1.f : -1.f; // square
		case 2: return std::sin(ph * 2.f * (float)M_PI); // sine
		case 3: {
			int steps = 8;
			int idx = std::max(0, std::min(steps - 1, (int)std::floor(ph * steps)));
			return lfoStepHash(idx + 17);
		}
		default: return 1.f - 2.f * ph; // saw down
	}
}

static void lfoDrawWaveTrace(int boxX, int boxY, int boxW, int boxH,
	int left, int right, float cy, float amp, int shape, float cycles, float phaseOffset)
{
	int prevY = 0;
	bool havePrev = false;
	for(int x = left; x <= right; ++x) {
		float f = (right == left) ? 0.f : (float)(x - left) / (float)(right - left);
		float ph = phaseOffset + f * cycles;
		float s = lfoSampleShape(shape, ph);
		int y = (int)std::lround(cy - s * amp);
		clippedPixel(boxX, boxY, boxW, boxH, x, y);
		if(havePrev) {
			if(y > prevY + 1) {
				for(int yy = prevY + 1; yy < y; ++yy)
					clippedPixel(boxX, boxY, boxW, boxH, x - 1, yy);
			} else if(y < prevY - 1) {
				for(int yy = prevY - 1; yy > y; --yy)
					clippedPixel(boxX, boxY, boxW, boxH, x - 1, yy);
			}
		}
		prevY = y;
		havePrev = true;
	}
}

static void lfoDrawCursor(int boxX, int boxY, int boxW, int boxH,
	int left, int right, float cy, float amp, int shape, float cycles, float phaseOffset, float cursorPh)
{
	cursorPh = std::max(0.f, std::min(1.f, cursorPh));
	int x = left + (int)std::lround(cursorPh * (float)(right - left));
	float ph = phaseOffset + cursorPh * cycles;
	float s = lfoSampleShape(shape, ph);
	int y = (int)std::lround(cy - s * amp);
	clippedLine(boxX, boxY, boxW, boxH, (float)x, y - 4.f, (float)x, y + 4.f, 0);
}

static void lfoDrawMiniRect(int boxX, int boxY, int boxW, int boxH, int x, int y, int w, int h, bool active)
{
	clippedLine(boxX, boxY, boxW, boxH, (float)x, (float)y, (float)(x + w - 1), (float)y, 0);
	clippedLine(boxX, boxY, boxW, boxH, (float)x, (float)(y + h - 1), (float)(x + w - 1), (float)(y + h - 1), 0);
	clippedLine(boxX, boxY, boxW, boxH, (float)x, (float)y, (float)x, (float)(y + h - 1), 0);
	clippedLine(boxX, boxY, boxW, boxH, (float)(x + w - 1), (float)y, (float)(x + w - 1), (float)(y + h - 1), 0);
	if(active && w > 6 && h > 6) {
		clippedLine(boxX, boxY, boxW, boxH, (float)(x + 2), (float)(y + 2), (float)(x + w - 3), (float)(y + 2), 0);
		clippedLine(boxX, boxY, boxW, boxH, (float)(x + 2), (float)(y + h - 3), (float)(x + w - 3), (float)(y + h - 3), 0);
		clippedLine(boxX, boxY, boxW, boxH, (float)(x + 2), (float)(y + 2), (float)(x + 2), (float)(y + h - 3), 0);
		clippedLine(boxX, boxY, boxW, boxH, (float)(x + w - 3), (float)(y + 2), (float)(x + w - 3), (float)(y + h - 3), 0);
	}
}

static void lfoDrawCenteredLabelInBox(int x, int y, int w, int h, const std::string& s)
{
	int tw = gOled.textWidth(s);
	int tx = x + std::max(1, (w - tw) / 2);
	int ty = y + std::max(1, (h - 7) / 2);
	gOled.drawText(tx, ty, s, true);
}

static void drawLfoVisual(int page, int selected, const float values[kParamCount], int boxX, int boxY, int boxW, int boxH)
{
	static const char* kBankNames[] = {"A", "B", "AB"};
	static const char* kTargetNames[] = {"NONE","MSTR","PITC","BRGT","POS","PICK","PART","IMPL","NCOL","L2RT","L2AM"};

	const int slot = lfoPageSlot(page);
	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;

	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	const int left = ix0 + 6;
	const int right = ix0 + iw - 7;
	const int top = iy0 + 4;
	const int bottom = iy0 + ih - 5;

	int shape = lfoShapeIndex(values[2]);

	switch(selected) {
		case 0: { // Target Bank
			int bankIdx = ((int)std::lround(values[0]) <= 1) ? 0 : (((int)std::lround(values[0]) == 2) ? 1 : 2);
			float pos = smoothVisualValue(gLfoSmooth[slot][0], (float)bankIdx);
			int w = 28;
			int h = 15;
			float spacing = 34.f;
			for(int i = 0; i < 3; ++i) {
				float bx = cx + ((float)i - pos) * spacing - w * 0.5f;
				int x = (int)std::lround(bx);
				int y = (int)std::lround(cy - h * 0.5f);
				bool active = std::fabs((float)i - pos) < 0.35f;
				lfoDrawMiniRect(boxX, boxY, boxW, boxH, x, y, w, h, active);
				lfoDrawCenteredLabelInBox(x, y, w, h, kBankNames[i]);
			}
		} break;

		case 1: { // Target carousel
			int targetCount = 11;
			int targetIdx = std::max(0, std::min(targetCount - 1, (int)std::lround(values[1])));
			float pos = smoothVisualValue(gLfoSmooth[slot][1], (float)targetIdx);
			int centerIdx = (int)std::lround(pos);
			int w = 34;
			int h = 15;
			float spacing = 38.f;
			for(int i = centerIdx - 1; i <= centerIdx + 1; ++i) {
				if(i < 0 || i >= targetCount)
					continue;
				float bx = cx + ((float)i - pos) * spacing - w * 0.5f;
				int x = (int)std::lround(bx);
				int y = (int)std::lround(cy - h * 0.5f);
				bool active = std::fabs((float)i - pos) < 0.35f;
				lfoDrawMiniRect(boxX, boxY, boxW, boxH, x, y, w, h, active);
				lfoDrawCenteredLabelInBox(x, y, w, h, centeredTrim(kTargetNames[i], 5));
			}
		} break;

		case 2: { // Shape
			float cursorPh = std::fmod(t * 0.32f, 1.f);
			lfoDrawWaveTrace(boxX, boxY, boxW, boxH, left, right, cy, ih * 0.24f, shape, 1.f, 0.f);
			lfoDrawCursor(boxX, boxY, boxW, boxH, left, right, cy, ih * 0.24f, shape, 1.f, 0.f, cursorPh);
		} break;

		case 3: { // Rate
			float rateN = lfoNormRate(values[3]);
			rateN = smoothVisualValue(gLfoSmooth[slot][3], rateN);
			float cycles = 1.f + rateN * 5.f;
			float cursorPh = std::fmod(t * (0.15f + rateN * 1.45f), 1.f);
			lfoDrawWaveTrace(boxX, boxY, boxW, boxH, left, right, cy, ih * 0.22f, shape, cycles, 0.f);
			lfoDrawCursor(boxX, boxY, boxW, boxH, left, right, cy, ih * 0.22f, shape, cycles, 0.f, cursorPh);
		} break;

		case 4: { // Mode
			int mode = lfoModeIndex(values[4]);
			float cursorPh = 0.f;
			if(mode == 0) {
				cursorPh = std::fmod(t * 0.55f, 1.f);
				clippedLine(boxX, boxY, boxW, boxH, (float)(left + 1), (float)(top + 1), (float)(left + 5), (float)(top + 1), 0);
				clippedLine(boxX, boxY, boxW, boxH, (float)(right - 5), (float)(top + 1), (float)(right - 1), (float)(top + 1), 0);
			} else {
				float cycle = std::fmod(t * 0.42f, 1.65f);
				cursorPh = (cycle < 1.f) ? cycle : 1.f;
				if(cursorPh > 0.02f) {
					int trailX = left + (int)std::lround(cursorPh * (float)(right - left));
					clippedLine(boxX, boxY, boxW, boxH, (float)left, (float)(bottom - 1), (float)trailX, (float)(bottom - 1), 0);
				}
			}
			lfoDrawWaveTrace(boxX, boxY, boxW, boxH, left, right, cy, ih * 0.22f, shape, 1.f, 0.f);
			lfoDrawCursor(boxX, boxY, boxW, boxH, left, right, cy, ih * 0.22f, shape, 1.f, 0.f, cursorPh);
		} break;

		case 5: { // Amount
			float amtRaw = std::max(-1.f, std::min(1.f, values[5]));
			float amt = smoothVisualValue(gLfoSmooth[slot][5], amtRaw);
			float mag = std::fabs(amt);
			float drawAmp = mag * (ih * 0.30f);
			satDrawHLine(boxX, boxY, boxW, boxH, left, right, (int)std::lround(cy));
			if(drawAmp > 0.25f)
				lfoDrawWaveTrace(boxX, boxY, boxW, boxH, left, right, cy, (amt < 0.f) ? -drawAmp : drawAmp, shape, 1.f, 0.f);
		} break;
	}
}


static void drawPerformanceVisual(int page, int selected, const float values[kParamCount], int boxX, int boxY, int boxW, int boxH)
{
	static const char* kBankNames[] = {"A", "B", "AB"};
	static const char* kPerfTargetNames[] = {"NONE","MSTR","BRGT","POS","PICK","ATK","DEC","REL","IMPL","NCOL","D1","D2","D3"};
	const int perfSlot = (page == kPageVelocity) ? 0 : 1;
	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;

	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	const int left = ix0 + 6;
	const int right = ix0 + iw - 7;
	const int top = iy0 + 4;
	const int bottom = iy0 + ih - 5;

	auto drawBankCarousel = [&]() {
		int bankIdx = ((int)std::lround(values[0]) <= 1) ? 0 : (((int)std::lround(values[0]) == 2) ? 1 : 2);
		float pos = smoothVisualValue(gPerfSmooth[perfSlot][0], (float)bankIdx);
		int w = 28;
		int h = 15;
		float spacing = 34.f;
		for(int i = 0; i < 3; ++i) {
			float bx = cx + ((float)i - pos) * spacing - w * 0.5f;
			int x = (int)std::lround(bx);
			int y = (int)std::lround(cy - h * 0.5f);
			bool active = std::fabs((float)i - pos) < 0.35f;
			lfoDrawMiniRect(boxX, boxY, boxW, boxH, x, y, w, h, active);
			lfoDrawCenteredLabelInBox(x, y, w, h, kBankNames[i]);
		}
	};

	auto drawTargetCarousel = [&]() {
		const int targetCount = 13;
		int targetIdx = std::max(0, std::min(targetCount - 1, (int)std::lround(values[1])));
		float pos = smoothVisualValue(gPerfSmooth[perfSlot][1], (float)targetIdx);
		int centerIdx = (int)std::lround(pos);
		int w = 34;
		int h = 15;
		float spacing = 38.f;
		for(int i = centerIdx - 1; i <= centerIdx + 1; ++i) {
			if(i < 0 || i >= targetCount)
				continue;
			float bx = cx + ((float)i - pos) * spacing - w * 0.5f;
			int x = (int)std::lround(bx);
			int y = (int)std::lround(cy - h * 0.5f);
			bool active = std::fabs((float)i - pos) < 0.35f;
			lfoDrawMiniRect(boxX, boxY, boxW, boxH, x, y, w, h, active);
			lfoDrawCenteredLabelInBox(x, y, w, h, centeredTrim(kPerfTargetNames[i], 5));
		}
	};

	auto drawVelocityAmount = [&]() {
		float amt = smoothVisualValue(gPerfSmooth[perfSlot][2], std::max(-1.f, std::min(1.f, values[2])));
		float mag = std::fabs(amt);
		float dir = (amt >= 0.f) ? -1.f : 1.f;
		float burstR = 1.5f + mag * 8.5f;
		float trail = 2.0f + mag * (ih * 0.28f);
		float hitY = cy;
		clippedLine(boxX, boxY, boxW, boxH, cx, hitY - 3.f, cx, hitY + 3.f, 0);
		clippedLine(boxX, boxY, boxW, boxH, cx, hitY, cx, hitY + dir * trail, mag > 0.72f ? 1 : 0);
		for(int i = 0; i < 6; ++i) {
			float a = (-1.5707963f * dir) + (-1.05f + i * 0.42f) * (0.38f + mag * 0.70f);
			float r = burstR * (0.72f + 0.10f * i);
			float x1 = cx + std::cos(a) * r;
			float y1 = hitY + std::sin(a) * r;
			clippedLine(boxX, boxY, boxW, boxH, cx, hitY, x1, y1, 0);
		}
		if(mag > 0.18f) {
			for(int i = 0; i < 3; ++i) {
				float yy = hitY + dir * (3.f + i * (2.0f + mag * 1.6f));
				float half = 1.f + i + mag * 2.5f;
				clippedLine(boxX, boxY, boxW, boxH, cx - half, yy, cx + half, yy, 0);
			}
		}
	};

	auto drawPressureAmount = [&]() {
		float amt = smoothVisualValue(gPerfSmooth[perfSlot][2], std::max(-1.f, std::min(1.f, values[2])));
		float mag = std::fabs(amt);
		float halfH = ih * 0.28f;
		float baseHalfW = 8.f;
		float halfW = (amt >= 0.f)
			? baseHalfW + mag * 7.f
			: std::max(2.5f, baseHalfW - mag * 5.5f);
		float topY = cy - halfH;
		float botY = cy + halfH;
		clippedLine(boxX, boxY, boxW, boxH, cx - halfW, topY, cx + halfW, topY, 0);
		clippedLine(boxX, boxY, boxW, boxH, cx - halfW, botY, cx + halfW, botY, 0);
		clippedLine(boxX, boxY, boxW, boxH, cx - halfW, topY, cx - halfW, botY, 0);
		clippedLine(boxX, boxY, boxW, boxH, cx + halfW, topY, cx + halfW, botY, 0);
		float innerHalfW = std::max(1.5f, halfW - 3.f);
		clippedLine(boxX, boxY, boxW, boxH, cx - innerHalfW, cy, cx + innerHalfW, cy, 0);
		if(amt >= 0.f) {
			for(int i = 0; i < 3; ++i) {
				float yy = cy + (i - 1) * (4.f + mag * 1.2f);
				float ex = halfW + 2.f + i * 2.f + mag * 2.f;
				clippedLine(boxX, boxY, boxW, boxH, cx - halfW, yy, cx - ex, yy, 0);
				clippedLine(boxX, boxY, boxW, boxH, cx + halfW, yy, cx + ex, yy, 0);
			}
		} else {
			for(int i = 0; i < 3; ++i) {
				float yy = cy + (i - 1) * 5.f;
				float ex = halfW + 4.f;
				clippedLine(boxX, boxY, boxW, boxH, cx - ex, yy, cx - halfW, yy, 0);
				clippedLine(boxX, boxY, boxW, boxH, cx + ex, yy, cx + halfW, yy, 0);
				clippedLine(boxX, boxY, boxW, boxH, cx - halfW - 2.f, yy - 1.f, cx - halfW, yy, 0);
				clippedLine(boxX, boxY, boxW, boxH, cx - halfW - 2.f, yy + 1.f, cx - halfW, yy, 0);
				clippedLine(boxX, boxY, boxW, boxH, cx + halfW + 2.f, yy - 1.f, cx + halfW, yy, 0);
				clippedLine(boxX, boxY, boxW, boxH, cx + halfW + 2.f, yy + 1.f, cx + halfW, yy, 0);
			}
		}
	};

	auto drawDeadZone = [&]() {
		float dz = smoothVisualValue(gPerfSmooth[perfSlot][3], std::max(0.f, std::min(1.f, values[3])));
		int laneW = 18;
		int laneX = (int)std::lround(cx - laneW * 0.5f);
		int laneY0 = top + 1;
		int laneY1 = bottom - 1;
		int laneH = std::max(8, laneY1 - laneY0);
		int threshY = laneY1 - (int)std::lround(dz * (float)laneH);
		lfoDrawMiniRect(boxX, boxY, boxW, boxH, laneX, laneY0, laneW, laneH + 1, false);
		for(int y = threshY + 1; y <= laneY1; ++y) {
			for(int x = laneX + 2; x <= laneX + laneW - 3; ++x) {
				if(((x + y) & 1) == 0)
					clippedPixel(boxX, boxY, boxW, boxH, x, y);
			}
		}
		clippedLine(boxX, boxY, boxW, boxH, (float)(laneX + 1), (float)threshY, (float)(laneX + laneW - 2), (float)threshY, 0);
		float activeMidX = laneX + laneW * 0.5f;
		clippedLine(boxX, boxY, boxW, boxH, activeMidX, (float)threshY, activeMidX, (float)(laneY0 + 2), 0);
	};

	auto drawPressureCurve = [&]() {
		float cv = smoothVisualValue(gPerfSmooth[perfSlot][4], std::max(-1.f, std::min(1.f, values[4])));
		float x0 = left + 2.f;
		float y0 = bottom - 1.f;
		float x1 = right - 1.f;
		float y1 = top + 1.f;
		clippedLine(boxX, boxY, boxW, boxH, x0, y0, x0, y1, 0);
		clippedLine(boxX, boxY, boxW, boxH, x0, y0, x1, y0, 0);
		const int segs = 48;
		float px = x0;
		float py = y0;
		for(int i = 1; i <= segs; ++i) {
			float u = (float)i / (float)segs;
			float shaped = (cv >= 0.f)
				? std::pow(u, 1.f + cv * 2.4f)
				: 1.f - std::pow(1.f - u, 1.f + (-cv) * 2.4f);
			float x = x0 + (x1 - x0) * u;
			float y = y0 + (y1 - y0) * shaped;
			clippedLine(boxX, boxY, boxW, boxH, px, py, x, y, 0);
			px = x;
			py = y;
		}
		float dotU = std::fmod(t * 0.24f, 1.f);
		float dotShaped = (cv >= 0.f)
			? std::pow(dotU, 1.f + cv * 2.4f)
			: 1.f - std::pow(1.f - dotU, 1.f + (-cv) * 2.4f);
		int dx = (int)std::lround(x0 + (x1 - x0) * dotU);
		int dy = (int)std::lround(y0 + (y1 - y0) * dotShaped);
		knockedOutPixel(boxX, boxY, boxW, boxH, dx, dy, 1);
	};

	switch(page) {
		case kPageVelocity:
			switch(selected) {
				case 0: drawBankCarousel(); break;
				case 1: drawTargetCarousel(); break;
				case 2: drawVelocityAmount(); break;
				default:
					clippedLine(boxX, boxY, boxW, boxH, (float)left, cy, (float)right, cy, 0);
					break;
			}
			break;

		case kPagePressure:
			switch(selected) {
				case 0: drawBankCarousel(); break;
				case 1: drawTargetCarousel(); break;
				case 2: drawPressureAmount(); break;
				case 3: drawDeadZone(); break;
				case 4: drawPressureCurve(); break;
				default:
					clippedLine(boxX, boxY, boxW, boxH, (float)left, cy, (float)right, cy, 0);
					break;
			}
			break;
	}
}


static void drawMacroPlaceholder(int boxX, int boxY, int boxW, int boxH, const char* label = "---")
{
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	int w = 28;
	int h = 15;
	int x = ix0 + (iw - w) / 2;
	int y = iy0 + (ih - h) / 2;
	lfoDrawMiniRect(boxX, boxY, boxW, boxH, x, y, w, h, false);
	lfoDrawCenteredLabelInBox(x, y, w, h, label);
}

static void drawMacroPositionBar(int boxX, int boxY, int boxW, int boxH, float u, bool pickup)
{
	u = std::max(0.f, std::min(1.f, u));
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float left = ix0 + 8.f;
	const float right = ix0 + iw - 9.f;
	const float cy = iy0 + ih * 0.5f;
	const float x = left + (right - left) * u;
	clippedLine(boxX, boxY, boxW, boxH, left, cy, right, cy, 0);
	clippedLine(boxX, boxY, boxW, boxH, left, cy - 4.f, left, cy + 4.f, 0);
	clippedLine(boxX, boxY, boxW, boxH, right, cy - 4.f, right, cy + 4.f, 0);
	if(pickup) {
		clippedHollowNode(boxX, boxY, boxW, boxH, (int)std::lround(x), (int)std::lround(cy));
		float ry = cy - 9.f;
		clippedLine(boxX, boxY, boxW, boxH, x, ry, x, cy - 3.f, 0);
		clippedLine(boxX, boxY, boxW, boxH, x - 3.f, ry, x + 3.f, ry, 0);
	} else {
		clippedLine(boxX, boxY, boxW, boxH, x, cy - 5.f, x, cy + 5.f, 0);
		clippedLine(boxX, boxY, boxW, boxH, x - 4.f, cy, x + 4.f, cy, 0);
		clippedLine(boxX, boxY, boxW, boxH, x - 3.f, cy - 3.f, x + 3.f, cy + 3.f, 0);
		clippedLine(boxX, boxY, boxW, boxH, x - 3.f, cy + 3.f, x + 3.f, cy - 3.f, 0);
		for(int i = 0; i < 3; ++i) {
			float rr = 4.f + i * 3.f;
			clippedCircle(boxX, boxY, boxW, boxH, x, cy, rr, 0, 0.65f);
		}
	}
}

static void drawGlobalBankVisual(const float values[kParamCount], int boxX, int boxY, int boxW, int boxH)
{
	static const char* kNames[] = {"A", "B"};
	const int slot = 2;
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	int bankIdx = ((int)std::lround(values[0]) <= 1) ? 0 : 1;
	float pos = smoothVisualValue(gMacroSmooth[slot][0], (float)bankIdx);
	int w = 28;
	int h = 15;
	float spacing = 34.f;
	for(int i = 0; i < 2; ++i) {
		float bx = cx + ((float)i - 0.5f - (pos - 0.5f)) * spacing - w * 0.5f;
		int x = (int)std::lround(bx);
		int y = (int)std::lround(cy - h * 0.5f);
		bool active = std::fabs((float)i - pos) < 0.35f;
		lfoDrawMiniRect(boxX, boxY, boxW, boxH, x, y, w, h, active);
		lfoDrawCenteredLabelInBox(x, y, w, h, kNames[i]);
	}
}

static void drawGlobalOctaveVisual(float value, int boxX, int boxY, int boxW, int boxH)
{
	const int slot = 2;
	float signedVal = std::max(-1.f, std::min(1.f, (std::fabs(value) <= 8.f) ? (value / 4.f) : (value / 48.f)));
	signedVal = smoothVisualValue(gMacroSmooth[slot][1], signedVal);
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	const int steps = 9;
	const float stepH = 3.f;
	const float stepW = 8.f;
	for(int i = 0; i < steps; ++i) {
		float rel = (float)i - (steps - 1) * 0.5f;
		float y = cy - rel * stepH;
		float width = stepW + std::fabs(rel) * 3.0f;
		clippedLine(boxX, boxY, boxW, boxH, cx - width, y, cx + width, y, 0);
	}
	float markerY = cy - signedVal * ((steps - 1) * 0.5f * stepH);
	float markerW = 18.f;
	clippedLine(boxX, boxY, boxW, boxH, cx - markerW, markerY, cx + markerW, markerY, 0);
	clippedLine(boxX, boxY, boxW, boxH, cx - markerW, markerY - 1.f, cx - markerW, markerY + 1.f, 0);
	clippedLine(boxX, boxY, boxW, boxH, cx + markerW, markerY - 1.f, cx + markerW, markerY + 1.f, 0);
}

static void drawGlobalSemitoneVisual(float value, int boxX, int boxY, int boxW, int boxH)
{
	const int slot = 2;
	float signedVal = std::max(-1.f, std::min(1.f, (std::fabs(value) <= 24.f) ? (value / 12.f) : (value / 48.f)));
	signedVal = smoothVisualValue(gMacroSmooth[slot][2], signedVal);
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	const int steps = 13;
	const float stepH = 2.0f;
	for(int i = 0; i < steps; ++i) {
		float rel = (float)i - (steps - 1) * 0.5f;
		float y = cy - rel * stepH;
		float width = 10.f + (std::fabs(rel) * 1.1f);
		clippedLine(boxX, boxY, boxW, boxH, cx - width, y, cx + width, y, 0);
	}
	float markerY = cy - signedVal * ((steps - 1) * 0.5f * stepH);
	clippedLine(boxX, boxY, boxW, boxH, cx - 20.f, markerY, cx + 20.f, markerY, 0);
	knockedOutPixel(boxX, boxY, boxW, boxH, (int)std::lround(cx), (int)std::lround(markerY), 1);
}

static void drawGlobalTuneVisual(float value, int boxX, int boxY, int boxW, int boxH)
{
	const int slot = 2;
	float signedVal = (std::fabs(value) <= 1.05f) ? std::max(-1.f, std::min(1.f, value)) : std::max(-1.f, std::min(1.f, value / 100.f));
	signedVal = smoothVisualValue(gMacroSmooth[slot][3], signedVal);
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cx = ix0 + iw * 0.5f;
	const float cy = iy0 + ih * 0.5f;
	const int left = ix0 + 8;
	const int right = ix0 + iw - 9;
	satDrawHLine(boxX, boxY, boxW, boxH, left, right, (int)std::lround(cy));
	clippedLine(boxX, boxY, boxW, boxH, cx, cy - 10.f, cx, cy + 10.f, 0);
	for(int i = -4; i <= 4; ++i) {
		float x = cx + i * 10.f;
		float h = (i == 0) ? 6.f : ((i & 1) ? 2.f : 4.f);
		clippedLine(boxX, boxY, boxW, boxH, x, cy - h, x, cy + h, 0);
	}
	float needleX = cx + signedVal * 34.f;
	clippedLine(boxX, boxY, boxW, boxH, needleX, cy - 12.f, needleX, cy + 12.f, 0);
	clippedLine(boxX, boxY, boxW, boxH, needleX - 3.f, cy - 9.f, needleX, cy - 12.f, 0);
	clippedLine(boxX, boxY, boxW, boxH, needleX + 3.f, cy - 9.f, needleX, cy - 12.f, 0);
}

static void drawPlayGlobalVisual(int page, int selected, const float values[kParamCount], int boxX, int boxY, int boxW, int boxH)
{
	const int playSlot = (page == kPagePlay) ? 0 : ((page == kPagePlayAlt) ? 1 : 2);
	(void)playSlot;
	switch(page) {
		case kPagePlay:
			switch(selected) {
				case 0: {
					float level = smoothVisualValue(gMacroSmooth[0][0], std::max(0.f, std::min(1.f, values[0])));
					const int ix0 = boxX + 1;
					const int iy0 = boxY + 1;
					const int iw = boxW - 2;
					const int ih = boxH - 2;
					const float cx = ix0 + iw * 0.5f;
					const float baseY = iy0 + ih - 4.f;
					float halfW = 3.f + level * 9.f;
					float topY = baseY - (5.f + level * (ih * 0.58f));
					clippedLine(boxX, boxY, boxW, boxH, cx - halfW, baseY, cx + halfW, baseY, 0);
					clippedLine(boxX, boxY, boxW, boxH, cx - halfW, topY, cx - halfW, baseY, 0);
					clippedLine(boxX, boxY, boxW, boxH, cx + halfW, topY, cx + halfW, baseY, 0);
					clippedLine(boxX, boxY, boxW, boxH, cx - halfW, topY, cx + halfW, topY, 0);
					for(int i = 1; i <= 3; ++i) {
						float yy = baseY - i * 6.f;
						float ww = halfW + i * (1.5f + level * 1.5f);
						if(yy > topY)
							clippedLine(boxX, boxY, boxW, boxH, cx - ww, yy, cx + ww, yy, 0);
					}
				} break;
				case 1: drawBlendStarNoiseVisual(values[1], boxX, boxY, boxW, boxH); break;
				case 2: drawBodyVisual(kPageBodyA1, 4, values[2], boxX, boxY, boxW, boxH); break;
				case 3: {
					float u = smoothVisualValue(gMacroSmooth[0][3], std::max(0.f, std::min(1.f, values[3])));
					drawMacroPositionBar(boxX, boxY, boxW, boxH, u, false);
				} break;
				case 4: {
					float u = smoothVisualValue(gMacroSmooth[0][4], std::max(0.f, std::min(1.f, values[4])));
					drawMacroPositionBar(boxX, boxY, boxW, boxH, u, true);
				} break;
				case 5: drawSpaceVisual(5, values[5], boxX, boxY, boxW, boxH); break;
				default: drawMacroPlaceholder(boxX, boxY, boxW, boxH); break;
			}
			break;

		case kPagePlayAlt:
			switch(selected) {
				case 0: drawBodyVisual(kPageBodyA1, 5, values[0], boxX, boxY, boxW, boxH); break;
				case 1: drawBodyVisual(kPageBodyA1, 0, values[1], boxX, boxY, boxW, boxH); break;
				case 2: drawBodyVisual(kPageBodyA1, 1, values[2], boxX, boxY, boxW, boxH); break;
				case 3: drawBodyVisual(kPageBodyA1, 2, values[3], boxX, boxY, boxW, boxH); break;
				case 4: drawBodyVisual(kPageBodyA1, 3, values[4], boxX, boxY, boxW, boxH); break;
				default: drawMacroPlaceholder(boxX, boxY, boxW, boxH); break;
			}
			break;

		case kPageGlobalEdit:
			switch(selected) {
				case 0: drawGlobalBankVisual(values, boxX, boxY, boxW, boxH); break;
				case 1: drawGlobalOctaveVisual(values[1], boxX, boxY, boxW, boxH); break;
				case 2: drawGlobalSemitoneVisual(values[2], boxX, boxY, boxW, boxH); break;
				case 3: drawGlobalTuneVisual(values[3], boxX, boxY, boxW, boxH); break;
				case 4: drawBodyVisual(kPageBodyA1, 5, values[4], boxX, boxY, boxW, boxH); break;
				default: drawMacroPlaceholder(boxX, boxY, boxW, boxH); break;
			}
			break;
	}
}


static void drawResonatorEditVisual(int selected, const float values[kParamCount], int boxX, int boxY, int boxW, int boxH)
{
	const uint64_t nowMs = uiNowMs();
	const float t = (float)nowMs * 0.001f;
	const int ix0 = boxX + 1;
	const int iy0 = boxY + 1;
	const int iw = boxW - 2;
	const int ih = boxH - 2;
	const float cy = iy0 + ih * 0.5f;
	const float left = ix0 + 6.f;
	const float right = ix0 + iw - 7.f;
	const float top = iy0 + 4.f;
	const float baseY = iy0 + ih - 4.f;
	const int totalModes = 16;

	auto smoothRes = [&](int idx, float target) -> float {
		idx = std::max(0, std::min(idx, kParamCount - 1));
		return smoothVisualValue(gMacroSmooth[2][idx], target);
	};

	auto normResIndex = [&](float v) -> float {
		if(std::fabs(v) <= 1.05f)
			return std::max(0.f, std::min(1.f, v));
		return norm01(v, 0.f, 31.f);
	};

	auto normResRatio = [&](float v) -> float {
		if(v <= 1.05f)
			return std::max(0.f, std::min(1.f, v));
		float clamped = std::max(0.125f, std::min(64.f, v));
		float lo = std::log(0.125f);
		float hi = std::log(64.f);
		return std::max(0.f, std::min(1.f, (std::log(clamped) - lo) / (hi - lo)));
	};

	auto normResGain = [&](float v) -> float {
		if(std::fabs(v) <= 1.05f)
			return std::max(0.f, std::min(1.f, v));
		return norm01(v, 0.f, 2.f);
	};

	auto normResDecay = [&](float v) -> float {
		if(v <= 1.05f)
			return std::max(0.f, std::min(1.f, v));
		return std::max(0.f, std::min(1.f, std::log1pf(std::max(0.f, v)) / std::log1pf(6000.f)));
	};

	float idx01 = smoothRes(0, normResIndex(values[0]));
	float ratio01 = smoothRes(1, normResRatio(values[1]));
	float gain01 = smoothRes(2, normResGain(values[2]));
	float decay01 = smoothRes(3, normResDecay(values[3]));
	int activeIdx = std::max(0, std::min(totalModes - 1, (int)std::lround(idx01 * (float)(totalModes - 1))));

	auto drawModeForest = [&](float highlightScale, bool haloed) {
		for(int i = 0; i < totalModes; ++i) {
			float fi = (totalModes <= 1) ? 0.5f : (float)i / (float)(totalModes - 1);
			float x = left + fi * (right - left);
			float ramp = 0.24f + 0.76f * std::pow(fi, 0.82f);
			float sway = std::sin(t * 1.6f + fi * 6.0f) * 1.2f;
			float h = 5.f + ramp * (ih * 0.45f) + sway;
			if(i == activeIdx)
				h *= highlightScale;
			float yTop = baseY - h;
			if(haloed && i == activeIdx)
				knockedOutLine(boxX, boxY, boxW, boxH, x, baseY, x, yTop, 0, 1);
			else
				clippedLine(boxX, boxY, boxW, boxH, x, baseY, x, yTop, 0);
			if(i == activeIdx) {
				knockedOutPixel(boxX, boxY, boxW, boxH, (int)std::lround(x), (int)std::lround(yTop), 1);
			}
		}
	};

	switch(selected) {
		case 0: { // Resonator index
			drawModeForest(1.18f, true);
			float x = left + idx01 * (right - left);
			clippedLine(boxX, boxY, boxW, boxH, x, top + 2.f, x, baseY + 1.f, 0);
			clippedLine(boxX, boxY, boxW, boxH, x - 4.f, top + 2.f, x + 4.f, top + 2.f, 0);
			clippedLine(boxX, boxY, boxW, boxH, x - 4.f, baseY + 1.f, x + 4.f, baseY + 1.f, 0);
		} break;

		case 1: { // Ratio
			const int steps = 8;
			float ladderX = left + 12.f;
			float markerX = right - 16.f;
			float topY = top + 2.f;
			float botY = baseY;
			clippedLine(boxX, boxY, boxW, boxH, ladderX, topY, ladderX, botY, 0);
			for(int i = 0; i < steps; ++i) {
				float fi = (steps <= 1) ? 0.5f : (float)i / (float)(steps - 1);
				float y = botY - fi * (botY - topY);
				float len = 4.f + (i & 1 ? 2.f : 0.f);
				clippedLine(boxX, boxY, boxW, boxH, ladderX - len, y, ladderX + len, y, 0);
			}
			float y = botY - ratio01 * (botY - topY);
			clippedLine(boxX, boxY, boxW, boxH, ladderX, y, markerX, y, 0);
			knockedOutPixel(boxX, boxY, boxW, boxH, (int)std::lround(markerX), (int)std::lround(y), 1);
			clippedLine(boxX, boxY, boxW, boxH, markerX, y - 4.f, markerX, y + 4.f, 0);
			float ghostY = botY - idx01 * (botY - topY);
			if(std::fabs(ghostY - y) > 1.5f)
				clippedLine(boxX, boxY, boxW, boxH, markerX - 7.f, ghostY, markerX - 2.f, ghostY, 0);
		} break;

		case 2: { // Gain
			drawModeForest(0.78f, false);
			float x = left + idx01 * (right - left);
			float h = 5.f + gain01 * (ih * 0.62f);
			float yTop = baseY - h;
			knockedOutLine(boxX, boxY, boxW, boxH, x, baseY, x, yTop, 1, 1);
			clippedLine(boxX, boxY, boxW, boxH, x - 4.f, yTop, x + 4.f, yTop, 0);
		} break;

		case 3: { // Decay
			float startX = left + 6.f;
			float endX = right - 2.f;
			float amp = 2.5f + gain01 * 6.0f;
			float survive = 0.18f + decay01 * 0.82f;
			float px = startX;
			float py = cy;
			clippedLine(boxX, boxY, boxW, boxH, startX - 5.f, cy, startX, cy, 0);
			clippedLine(boxX, boxY, boxW, boxH, startX - 3.f, cy - 3.f, startX, cy, 0);
			clippedLine(boxX, boxY, boxW, boxH, startX - 3.f, cy + 3.f, startX, cy, 0);
			const int n = 96;
			for(int i = 1; i <= n; ++i) {
				float fi = (float)i / (float)n;
				float x = startX + fi * (endX - startX);
				float env = std::exp(-fi * (2.4f - survive * 2.0f));
				float y = cy + std::sin(fi * (10.f + decay01 * 16.f) * (float)M_PI + t * 3.0f) * amp * env;
				clippedLine(boxX, boxY, boxW, boxH, px, py, x, y, 0);
				px = x;
				py = y;
			}
			for(int k = 1; k <= 3; ++k) {
				float fx = startX + survive * (endX - startX) * ((float)k / 3.f);
				float rr = 1.5f + k * 1.8f;
				clippedCircle(boxX, boxY, boxW, boxH, fx, cy, rr, 0, 0.7f);
			}
		} break;

		default:
			drawMacroPlaceholder(boxX, boxY, boxW, boxH);
			break;
	}
}

static void drawParamOverlay(int page, int selected, const float values[kParamCount])
{
	std::string name = centeredTrim(fullParamName(page, selected), 20);
	std::string value = formatValueOverlay(page, selected, values[selected]);

	const int stripX = 3;
	const int stripW = 122;
	const int topStripH = 11;
	const int bottomStripH = 11;
	const int topY = 1;
	const int bottomY = 51;
	const int centerX = 6;
	const int centerY = 14;
	const int centerW = 116;
	const int centerH = 35;

	gOled.clear();
	gOled.fillRect(stripX, topY, stripW, topStripH, true);
	gOled.rect(stripX, topY, stripW, topStripH, true);
	drawCenteredText(topY + 2, name, false);

	gOled.rect(centerX, centerY, centerW, centerH, true);
	if(isBodyPage(page))
		drawBodyVisual(page, selected, values[selected], centerX, centerY, centerW, centerH);
	else if(page == kPageDampers)
		drawDamperVisual(selected, values[selected], centerX, centerY, centerW, centerH);
	else if(page == kPageExciterA || page == kPageExciterB)
		drawExciterVisual(page, selected, values[selected], centerX, centerY, centerW, centerH);
	else if(page == kPageSpace)
		drawSpaceVisual(selected, values[selected], centerX, centerY, centerW, centerH);
	else if(page == kPageEcho)
		drawEchoVisual(selected, values, centerX, centerY, centerW, centerH);
	else if(page == kPageSaturation)
		drawSaturationVisual(selected, values[selected], centerX, centerY, centerW, centerH);
	else if(page == kPageModLfo1 || page == kPageModLfo2)
		drawLfoVisual(page, selected, values, centerX, centerY, centerW, centerH);
	else if(page == kPageVelocity || page == kPagePressure)
		drawPerformanceVisual(page, selected, values, centerX, centerY, centerW, centerH);
	else if(page == kPageResonatorEdit)
		drawResonatorEditVisual(selected, values, centerX, centerY, centerW, centerH);
	else if(page == kPagePlay || page == kPagePlayAlt || page == kPageGlobalEdit)
		drawPlayGlobalVisual(page, selected, values, centerX, centerY, centerW, centerH);

	gOled.rect(stripX, bottomY, stripW, bottomStripH, true);
	drawCenteredText(bottomY + 2, value, true);
	gOled.display();
}

static void drawScreenSnapshot(int page, int selected, int presetSlot, const float values[kParamCount])
{
	uint64_t nowMs = uiNowMs();

	bool overlayVisible = false;
	int overlayParam = gOverlayParam;
	if(isOverlayEligiblePage(page) && gOverlayPage == page && overlayParam >= 0 && overlayParam < kParamCount) {
		if(gOverlayUntilMs > nowMs)
			overlayVisible = true;
		else {
			gOverlayUntilMs = 0;
			gOverlayPage = -1;
			gOverlayParam = -1;
			overlayParam = -1;
		}
	}

	if(overlayVisible) {
		drawParamOverlay(page, overlayParam, values);
		return;
	}

	gOled.clear();

	gOled.fillRect(0, 0, 128, 10, true);
	std::string title = pageName(page);
	if(gScreenState.patchDirty) title += "*";
	gOled.drawText(2, 1, title, false);
	gOled.hLine(0, 127, 11, true);

	if(page == kPagePreset) {
		char slotBuf[24];
		snprintf(slotBuf, sizeof(slotBuf), gScreenState.patchDirty ? "P%02d*" : "P%02d", presetSlot + 1);
		std::string modeText = formatValue(page, 1, values[1]);
		std::string usedText = formatValue(page, 2, values[2]);

		// Top status row: slot / mode / used-empty, each in its own box so text never overlaps.
		gOled.rect(0, 14, 32, 12, true);
		gOled.drawText(5, 17, slotBuf, true);
		gOled.rect(36, 14, 40, 12, true);
		gOled.drawText(40, 17, modeText, true);
		gOled.rect(80, 14, 48, 12, true);
		if(usedText == "EMPTY")
			gOled.drawText(83, 17, usedText, true);
		else
			gOled.drawText(86, 17, usedText, true);

		std::string name(gScreenState.presetName, gScreenState.presetName + 16);
		gOled.rect(0, 31, 128, 14, true);
		gOled.drawText(4, 35, name, true);

		int cur = std::max(0, std::min(gScreenState.presetCursor, 15));
		if(gScreenState.presetMode == 1) {
			int cx = 4 + cur * 6;
			gOled.hLine(cx, cx + 4, 46, true);
		}

		std::string footer;
		if(gScreenState.feedback == 1) footer = "LOADED";
		else if(gScreenState.feedback == 2) footer = "SAVED";
		else if(gScreenState.feedback == 3) footer = "OVERWRITE";
		else if(gScreenState.feedback == 4) footer = "REVERTED";
		else if(gScreenState.presetMode == 0)
			footer = gScreenState.patchDirty ? "ENC SKIP BK REV" : "PRESS LOAD";
		else if(gScreenState.presetMode == 1)
			footer = "SAVE>NXT CUR " + std::to_string(cur + 1);
		else
			footer = "PRESS SAVE";
		gOled.drawText(2, 54, footer, true);
		gOled.display();
		return;
	}

	const char* labels[kParamCount];
	pageLabels(page, labels);

	const int rowY[3] = {17, 31, 45};
	for(int row = 0; row < 3; ++row) {
		int left = row * 2;
		int right = left + 1;

		std::string lText = std::string(labels[left]) + " " + formatValue(page, left, values[left]);
		std::string rText = std::string(labels[right]) + " " + formatValue(page, right, values[right]);

		if(selected == left) {
			gOled.fillRect(0, rowY[row] - 1, 61, 9, true);
			gOled.drawText(2, rowY[row], lText, false);
		} else {
			gOled.drawText(2, rowY[row], lText, true);
		}

		if(selected == right) {
			gOled.fillRect(66, rowY[row] - 1, 61, 9, true);
			gOled.drawText(68, rowY[row], rText, false);
		} else {
			gOled.drawText(68, rowY[row], rText, true);
		}
	}

	gOled.display();
}

static void oledTask(void*)
{
	using Clock = std::chrono::steady_clock;
	auto lastDraw = Clock::now();
	while(!Bela_stopRequested()) {
		bool needsDraw = false;
		if(gOledReady.load()) {
			if(gScreenState.dirty.exchange(false))
				needsDraw = true;
			else if(std::chrono::duration_cast<std::chrono::milliseconds>(Clock::now() - lastDraw).count() >= 80)
				needsDraw = true;
		}
		if(needsDraw) {
			int page = 0;
			int selected = 0;
			int presetSlot = 0;
			float values[kParamCount];
			{
				std::lock_guard<std::mutex> lock(gScreenState.mutex);
				page = gScreenState.page;
				selected = gScreenState.selected;
				presetSlot = gScreenState.presetSlot;
				for(int i = 0; i < kParamCount; ++i)
					values[i] = gScreenState.values[i];
			}
			drawScreenSnapshot(page, selected, presetSlot, values);
			lastDraw = Clock::now();
		}
		usleep(30000);
	}
}

// ------------------------------------------------------------
// Pd float hook
// ------------------------------------------------------------
int floatHook(const char *source, float value)
{
	std::lock_guard<std::mutex> lock(gScreenState.mutex);

	if(strcmp(source, "bela_screen_page") == 0) {
		int newPage = std::max(0, std::min((int)std::round(value), kPageCount - 1));
		if(newPage != gScreenState.page) {
			gScreenState.page = newPage;
			gIgnoreParamUntilMs = uiNowMs() + 180;
			for(auto& t : gParamChangedMs) t = 0;
			gOverlayUntilMs = 0;
			gOverlayPage = -1;
			gOverlayParam = -1;
		} else {
			gScreenState.page = newPage;
		}
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_selected") == 0) {
		int newSel = std::max(-1, std::min((int)std::round(value), kParamCount - 1));
		gScreenState.selected = newSel;
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_preset_slot") == 0) {
		gScreenState.presetSlot = std::max(0, (int)std::round(value));
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_preset_mode") == 0) {
		gScreenState.presetMode = (int)std::round(value);
		gScreenState.values[1] = value;
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_preset_cursor") == 0) {
		gScreenState.presetCursor = std::max(0, std::min((int)std::round(value), 15));
		gScreenState.values[3] = value;
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_preset_used") == 0) {
		gScreenState.presetUsed = value > 0.5f;
		gScreenState.values[2] = value;
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_feedback") == 0) {
		gScreenState.feedback = (int)std::round(value);
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_patch_dirty") == 0) {
		gScreenState.patchDirty = value > 0.5f;
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_touch") == 0) {
		int nonce = (int)std::round(value);
		if(nonce != gOverlayTouchNonce) {
			gOverlayTouchNonce = nonce;
			if(isOverlayEligiblePage(gScreenState.page) && gScreenState.selected >= 0 && gScreenState.selected < kParamCount) {
				gOverlayPage = gScreenState.page;
				gOverlayParam = gScreenState.selected;
				gOverlayUntilMs = uiNowMs() + 320;
			} else {
				gOverlayUntilMs = 0;
				gOverlayPage = -1;
				gOverlayParam = -1;
			}
		}
		gScreenState.dirty = true;
		return 0;
	}
		if(strncmp(source, "bela_screen_preset_name", 23) == 0) {
		int idx = source[23] - '0';
		if(idx >= 0 && idx < 8) {
			int packed = (int)std::round(value);
			gScreenState.presetName[idx * 2] = (char)(packed & 0xFF);
			gScreenState.presetName[idx * 2 + 1] = (char)((packed >> 8) & 0xFF);
			gScreenState.presetName[16] = '\0';
			gScreenState.dirty = true;
			return 0;
		}
	}
	if(strcmp(source, "bela_screen_param0") == 0) {
		float oldv = gScreenState.values[0];
		gScreenState.values[0] = value;
		if(gScreenState.page == kPageExciterA) gExciterACache[0] = value;
		if(gScreenState.page == kPageExciterB) gExciterBCache[0] = value;
		noteParamDisplayChangeLocked(0, oldv, value);
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_param1") == 0) {
		float oldv = gScreenState.values[1];
		gScreenState.values[1] = value;
		if(gScreenState.page == kPageExciterA) gExciterACache[1] = value;
		if(gScreenState.page == kPageExciterB) gExciterBCache[1] = value;
		noteParamDisplayChangeLocked(1, oldv, value);
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_param2") == 0) {
		float oldv = gScreenState.values[2];
		gScreenState.values[2] = value;
		if(gScreenState.page == kPageExciterA) gExciterACache[2] = value;
		if(gScreenState.page == kPageExciterB) gExciterBCache[2] = value;
		noteParamDisplayChangeLocked(2, oldv, value);
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_param3") == 0) {
		float oldv = gScreenState.values[3];
		gScreenState.values[3] = value;
		if(gScreenState.page == kPageExciterA) gExciterACache[3] = value;
		if(gScreenState.page == kPageExciterB) gExciterBCache[3] = value;
		noteParamDisplayChangeLocked(3, oldv, value);
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_param4") == 0) {
		float oldv = gScreenState.values[4];
		gScreenState.values[4] = value;
		if(gScreenState.page == kPageExciterA) gExciterACache[4] = value;
		if(gScreenState.page == kPageExciterB) gExciterBCache[4] = value;
		noteParamDisplayChangeLocked(4, oldv, value);
		gScreenState.dirty = true;
		return 0;
	}
	if(strcmp(source, "bela_screen_param5") == 0) {
		float oldv = gScreenState.values[5];
		gScreenState.values[5] = value;
		if(gScreenState.page == kPageExciterA) gExciterACache[5] = value;
		if(gScreenState.page == kPageExciterB) gExciterBCache[5] = value;
		noteParamDisplayChangeLocked(5, oldv, value);
		gScreenState.dirty = true;
		return 0;
	}

	return 1;
}

void Bela_userSettings(BelaInitSettings *settings)
{
	settings->uniformSampleRate = 1;
	settings->interleave = 0;
	settings->analogOutputsPersist = 0;
}

bool setup(BelaContext *context, void *userData)
{
	BelaLibpdSettings s{};
	s.floatHook = floatHook;
	s.bindSymbols = {
		"bela_screen_page",
		"bela_screen_selected",
		"bela_screen_preset_slot",
		"bela_screen_preset_mode",
		"bela_screen_preset_cursor",
		"bela_screen_preset_used",
		"bela_screen_feedback",
		"bela_screen_patch_dirty",
		"bela_screen_touch",
		"bela_screen_preset_name0",
		"bela_screen_preset_name1",
		"bela_screen_preset_name2",
		"bela_screen_preset_name3",
		"bela_screen_preset_name4",
		"bela_screen_preset_name5",
		"bela_screen_preset_name6",
		"bela_screen_preset_name7",
		"bela_screen_param0",
		"bela_screen_param1",
		"bela_screen_param2",
		"bela_screen_param3",
		"bela_screen_param4",
		"bela_screen_param5"
	};

	bool success = BelaLibpd_setup(context, userData, s);
	if(!success)
		return false;

	if(!gOled.setup(kI2CBus, kI2CAddress)) {
		rt_fprintf(stderr, "OLED setup failed\n");
	} else {
		gOledReady = true;
		gOledTask = Bela_createAuxiliaryTask(oledTask, 40, "oled-task");
		Bela_scheduleAuxiliaryTask(gOledTask);
	}

	return true;
}

void render(BelaContext *context, void *userData)
{
	BelaLibpd_render(context, userData);
}

void cleanup(BelaContext *context, void *userData)
{
	(void)context;
	(void)userData;
	gOled.cleanup();
	BelaLibpd_cleanup(context, userData);
}
