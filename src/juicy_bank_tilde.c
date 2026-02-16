// juicy_bank~ — modal resonator bank (V5.0)
// 6-voice poly (no tail-voice logic), true stereo banks, Behavior + Body + Individual inlets.
// NEW (V5.0):
//   • **Spacing** inlet (after dispersion, before anisotropy): nudges each mode toward the *next* harmonic
//     ratio (ceil or +1 if already integer). 0 = no shift, 1 = fully at next ratio.
//   • **32 modes by default**: startup ratios 1..32, gain=1.0, decay=1000 ms, attack=0, curve=0 (linear).
//   • **ZDF SVF resonators (TPT / Zavalishin)**: biquad/2-pole recursion replaced by stable
//     topology-preserving state-variable filters (bandpass output). Old resonator normalizers are removed.
//
// Build (macOS):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type 
//     -I"/Applications/Pd-0.56-1.app/Contents/Resources/src" 
//     -arch arm64 -arch x86_64 -mmacosx-version-min=10.13 
//     -bundle -undefined dynamic_lookup 
//     -o juicy_bank~.pd_darwin juicy_bank_tilde.c
//
// Build (Linux):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type 
//     -I"/usr/include/pd" -shared -fPIC -Wl,-export-dynamic -lm 
//     -o juicy_bank~.pd_linux juicy_bank_tilde.c

#include "m_pd.h"

/*
  Build/CPU notes (especially for Bela / ARMv7):
  - Use aggressive optimization flags in your build system, e.g.
      -O3 (or -Ofast) -ffast-math -fno-math-errno
      -mfpu=neon -mfloat-abi=hard (ARMv7 hard-float)
  - Avoid any Pd API calls (gensym/post/pd_error/memory alloc) in perform().
*/
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>

// ---------- Optional ARM NEON SIMD (ARMv7) ----------
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
  #include <arm_neon.h>
  #define JB_HAVE_NEON 1
#else
  #define JB_HAVE_NEON 0
#endif

#ifndef JB_ENABLE_NEON
  // Default: enable on NEON-capable builds, can be disabled via -DJB_ENABLE_NEON=0
  #define JB_ENABLE_NEON 1
#endif



// Denormal/subnormal protection (prevents CPU spikes on Intel when signals decay to tiny values)
#if defined(__SSE__) || defined(__SSE2__)
#include <xmmintrin.h>
  #if defined(__SSE3__)
  #include <pmmintrin.h>
  #endif
#endif
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- limits ----------
#define JB_GLOBAL_DECAY_MIN_S 0.02f   /* global decay floor (seconds) */
#define JB_GLOBAL_DECAY_MAX_S 10.0f   /* global decay ceiling (seconds) */
#define JB_DAMP_T60_MIN_S     0.005f  /* high-frequency minimum T60 at damper=1 (seconds) */

#define JB_MAX_MODES    32
#define JB_MAX_VOICES    6
#define JB_ATTACK_VOICES 6
#define JB_N_MODSRC 4
#define JB_N_MODTGT    15
#define JB_N_LFO       2
#define JB_N_DAMPERS  3
#define JB_PITCH_MOD_SEMITONES  2.0f

#define JB_ENABLE_FEEDBACK 0
#define JB_STEAL_FADE_SAMPLES 64

// ---------- SPACE (global stereo room) ----------
#define JB_SPACE_NCOMB     8
#define JB_SPACE_NCOMB_CH  4
#define JB_SPACE_MAX_DELAY 1700
#define JB_SPACE_NAP       4
#define JB_SPACE_AP_MAX    700

// Feedback delay line (per voice). Power-of-two ring buffer for fast wrap.



// ---------- utils ----------
static inline float jb_clamp(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }

// 3rd-order (4-point) Lagrange interpolation for fractional delay reads.
// w is the write index pointing to the NEXT sample to be written.
// 3rd-order (4-point) Lagrange interpolation read with PRECOMPUTED coefficients.
// This avoids floorf() + coefficient recompute in the hot per-sample loop.
// intDelay = floor(delay_samps), frac = delay_samps - intDelay (0..1).
// If frac==0, set frac_is_zero=1 and pass i0_offset=intDelay, else i0_offset=intDelay+1.
// 1st-order allpass (phase dispersion)
static inline float jb_allpass1_run(float x, float a, float *x1, float *y1){
    const float y = (a * x) + (*x1) - (a * (*y1));
    *x1 = x;
    *y1 = y;
    return y;
}
static inline float jb_wrap01(float x){
    x = x - floorf(x);
    if (x < 0.f) x += 1.f;
    return x;
}

static inline float jb_expmap01(float t, float lo, float hi){
    // Exponential mapping for better knob resolution at short times.
    if (t <= 0.f) return lo;
    if (t >= 1.f) return hi;
    if (lo <= 0.f) return lo + t*(hi-lo);
    return lo * powf(hi/lo, t);
}
static inline float jb_slope_to_powerlaw(float slope01){
    float s = jb_clamp(slope01, 0.f, 1.f);
    // Anchor points:
    //   s=0.0 -> 1.0 (linear)
    //   s=0.5 -> 2.0 (quadratic)
    //   s=1.0 -> 8.0 (extreme)
    if (s <= 0.5f){
        return 1.f + (s / 0.5f) * (2.f - 1.f);
    } else {
        return 2.f + ((s - 0.5f) / 0.5f) * (8.f - 2.f);
    }
}
// Velocity mapping: finite toggle set (unlimited simultaneous targets via toggles).
// All enabled targets share the same velmap_amount scaling.
// Toggle behavior: sending the same target symbol again flips it off.
typedef enum {
    JB_VEL_BELL_Z_D1_B1 = 0,
    JB_VEL_BELL_Z_D1_B2,
    JB_VEL_BELL_Z_D2_B1,
    JB_VEL_BELL_Z_D2_B2,
    JB_VEL_BELL_Z_D3_B1,
    JB_VEL_BELL_Z_D3_B2,

    JB_VEL_BRIGHTNESS_1,
    JB_VEL_BRIGHTNESS_2,
    JB_VEL_POSITION_1,
    JB_VEL_POSITION_2,
    JB_VEL_PICKUP_1,
    JB_VEL_PICKUP_2,
    JB_VEL_MASTER_1,
    JB_VEL_MASTER_2,

    JB_VEL_ADSR_ATTACK,
    JB_VEL_ADSR_DECAY,
    JB_VEL_ADSR_RELEASE,

    JB_VEL_IMP_SHAPE,
    JB_VEL_NOISE_TIMBRE,

    JB_VELMAP_N_TARGETS
} jb_velmap_idx;



// -------------------- PRESET SYSTEM (memory-only, 64 slots) --------------------
#define JB_PRESET_SLOTS        64
#define JB_PRESET_NAME_MAX     16

typedef enum {
    JB_PRESET_MODE_NORMAL = 0,
    JB_PRESET_MODE_NAMING = 1,
    JB_PRESET_MODE_SLOT   = 2
} jb_preset_mode_t;

// single-preset snapshot (add fields as synth grows; kept as plain floats for speed)
typedef struct _jb_preset {
    int   used;                         // 0=empty
    char  name[JB_PRESET_NAME_MAX + 1]; // null-terminated

    // bank globals
    float bank_master[2];
    int   bank_semitone[2];
    int   bank_octave[2];
    float bank_tune_cents[2];

    float release_amt[2];
    float stretch[2];
    float warp[2];
    float brightness[2];
    float density_amt[2];
    int   density_mode[2];
    float dispersion[2];
    float odd_skew[2];
    float even_skew[2];
    float collision_amt[2];
    float micro_detune[2];

    // spatial positions
    float excite_pos[2];
    float pickup_pos[2];

    // bell dampers (zeta basis params)
    float bell_peak_hz[2][JB_N_DAMPERS];
    float bell_peak_zeta_param[2][JB_N_DAMPERS];
    float bell_npl[2][JB_N_DAMPERS];
    float bell_npr[2][JB_N_DAMPERS];
    float bell_npm[2][JB_N_DAMPERS];

    // SPACE
    float space_size, space_decay, space_diffusion, space_damping, space_wetdry;

    // exciter
    float exc_fader;
    float exc_attack_ms, exc_decay_ms, exc_sustain, exc_release_ms;
    float exc_density;
    float exc_imp_shape;
    float exc_shape;

    // LFOs
    float lfo_index; // 1..2
    float lfo_shape_v[JB_N_LFO];
    float lfo_rate_v[JB_N_LFO];
    float lfo_phase_v[JB_N_LFO];
    float lfo_mode_v[JB_N_LFO];
    float lfo_amt_v[JB_N_LFO];
    t_symbol *lfo_target[JB_N_LFO];

    // velocity mapping
    float   velmap_amount;
    uint8_t velmap_on[JB_VELMAP_N_TARGETS];
} jb_preset_t;

typedef struct { unsigned int s; } jb_rng_t;
static inline void jb_rng_seed(jb_rng_t *r, unsigned int s){ if(!s) s=1; r->s = s; }
static inline unsigned int jb_rng_u32(jb_rng_t *r){ unsigned int x = r->s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; r->s = x; return x; }
static inline float jb_rng_uni(jb_rng_t *r){ return (jb_rng_u32(r) >> 8) * (1.0f/16777216.0f); }
static inline float jb_rng_bi(jb_rng_t *r){ return 2.f * jb_rng_uni(r) - 1.f; }

// ---------- safety / stability ----------
// If a voice ever produces NaN/INF or astronomically large values (runaway), we hard-reset that voice
// to prevent audio-thread stalls / "freezing".
// Threshold is extremely high so it won't affect normal audio.
#ifndef JB_PANIC_ABS_MAX
#define JB_PANIC_ABS_MAX 1.0e6f
#endif

static inline int jb_isfinitef(float x){
    return isfinite(x);
}
static inline float jb_kill_denorm(float x){
    return (fabsf(x) < 1e-20f) ? 0.f : x;
}

// ---------- ZDF State Variable Filter (SVF) resonators (Topology-Preserving Transform, Zavalishin style) ----------
// We replace the old 2-pole recursion with a ZDF SVF tick (TPT discretization).
// Notation (Vadim Zavalishin):
//   g = tan(pi * f / Fs)
//   R = damping parameter
//   d = 1 / (1 + 2*R*g + g*g)
// Tick (bandpass output):
//   v1 = (s1 + g*(x - s2)) * d
//   v2 = s2 + g*v1
//   s1 = 2*v1 - s1
//   s2 = 2*v2 - s2
typedef struct {
    float g, R, d;
    float s1, s2;
} jb_svf_t;

static inline void jb_svf_reset(jb_svf_t *f){
    f->g = f->R = f->d = 0.f;
    f->s1 = f->s2 = 0.f;
}

static inline void jb_svf_set_params(jb_svf_t *f, float g, float R){
    if (!isfinite(g) || g < 0.f) g = 0.f;
    if (!isfinite(R) || R < 0.f) R = 0.f;
    // our damper provides sane values, but clamp as a last resort
    if (R > 2.0f) R = 2.0f;
    f->g = g;
    f->R = R;
    float denom = 1.f + 2.f * R * g + g * g;
    if (!isfinite(denom) || denom <= 1e-20f) denom = 1e-20f;
    f->d = 1.f / denom;
}

static inline float jb_svf_bp_tick(jb_svf_t *f, float x){
    const float g = f->g;
    const float d = f->d;
    if (g == 0.f || d == 0.f) return 0.f;
    const float v1 = (f->s1 + g * (x - f->s2)) * d;
    const float v2 = f->s2 + g * v1;
    f->s1 = jb_kill_denorm(2.f * v1 - f->s1);
    f->s2 = jb_kill_denorm(2.f * v2 - f->s2);
    return v1; // bandpass
}

#if JB_HAVE_NEON && JB_ENABLE_NEON
// 4-lane ZDF SVF bandpass tick (NEON). Updates s1/s2 in-place.
// Equations match jb_svf_bp_tick() lane-wise.
static inline float32x4_t jb_svf_bp_tick4(float32x4_t g, float32x4_t d, float32x4_t *s1, float32x4_t *s2, float32x4_t x){
    // v1 = (s1 + g*(x - s2)) * d
    float32x4_t v1 = vmlaq_f32(*s1, g, vsubq_f32(x, *s2));
    v1 = vmulq_f32(v1, d);
    // v2 = s2 + g*v1
    float32x4_t v2 = vmlaq_f32(*s2, g, v1);
    // s1 = 2*v1 - s1
    *s1 = vsubq_f32(vaddq_f32(v1, v1), *s1);
    // s2 = 2*v2 - s2
    *s2 = vsubq_f32(vaddq_f32(v2, v2), *s2);
    return v1; // bandpass output
}

// Horizontal add helper compatible with ARMv7 (no vaddvq on older toolchains)
static inline float jb_hadd_f32x4(float32x4_t v){
    float32x2_t lo = vget_low_f32(v);
    float32x2_t hi = vget_high_f32(v);
    float32x2_t sum = vadd_f32(lo, hi);
    sum = vpadd_f32(sum, sum);
    return vget_lane_f32(sum, 0);
}
#endif


// ---------- INTERNAL EXCITER (Fusion STEP 1) ----------
// This is the former juicy_exciter~ DSP engine embedded into juicy_bank~.
// STEP 1: adds exciter DSP structs + helpers + per-voice exciter state storage + param inlets.
// STEP 2: removes external exciter audio inlets, runs the exciter per voice, injects stereo
//         exciter into BOTH banks (pre-modal injection), and feeds per-voice env into mod matrix.

#define JB_EXC_NVOICES   JB_MAX_VOICES
// Noise diffusion (all-pass) + color slope constants
#define JB_EXC_SLOPE_PIVOT_HZ  1000.f
#define JB_EXC_COLOR_OCT_SPAN  3.f   // approx octaves from pivot to spectral edge

// ---------- SPACE (Schroeder-style reverb) ----------
static const int jb_space_base_delay[JB_SPACE_NCOMB] = { 1117, 1373, 1481, 1607, 1103, 1361, 1471, 1597 };
static const int jb_space_ap_delay[JB_SPACE_NAP]     = { 225, 556, 341, 441 }; // L:225,556 | R:341,441 (primes)

static inline float jb_space_comb_tick(float *buf, int maxlen, int *w, int delay,
                                       float in, float g, float damp, float *lp_state)
{
    int wi = *w;
    int ri = wi - delay;
    if (ri < 0) ri += maxlen;

    float y = buf[ri];

    // 1-pole LP on FEEDBACK ONLY (damping): fb = (1-d)*y + d*lp_prev
    float fb = (1.f - damp) * y + damp * (*lp_state);
    *lp_state = fb;

    buf[wi] = in + g * fb;

    wi++;
    if (wi >= maxlen) wi = 0;
    *w = wi;
    // Return the undamped comb output; damping only affects the loop.
    return y;
}

// Classic Schroeder allpass: y = -g*x + z; z = x + g*y
static inline float jb_space_ap_tick(float *buf, int maxlen, int *w, int delay,
                                     float x, float g)
{
    int wi = *w;
    int ri = wi - delay;
    if (ri < 0) ri += maxlen;

    float z = buf[ri];
    float y = -g * x + z;
    buf[wi] = x + g * y;

    wi++;
    if (wi >= maxlen) wi = 0;
    *w = wi;
    return y;
}



#define JB_EXC_AP1_BASE 211
#define JB_EXC_AP2_BASE 503
#define JB_EXC_AP3_BASE 883
#define JB_EXC_AP4_BASE 1217
// max delay = base * 2.0 (tscale max) + 8 safety
#define JB_EXC_AP1_MAX (JB_EXC_AP1_BASE*2 + 8)
#define JB_EXC_AP2_MAX (JB_EXC_AP2_BASE*2 + 8)
#define JB_EXC_AP3_MAX (JB_EXC_AP3_BASE*2 + 8)
#define JB_EXC_AP4_MAX (JB_EXC_AP4_BASE*2 + 8)

static inline float jb_exc_expmap01(float t, float lo, float hi){
    if (t <= 0.f) return lo;
    if (t >= 1.f) return hi;
    return lo * powf(hi/lo, t);
}
static inline float jb_exc_midi_to_vel01(float v){
    // Accept either normalized 0..1 or MIDI-style 0..127.
    // NOTE: MIDI velocities can be 1 or 2 (very soft) — do NOT treat those as "full scale".
    if (v <= 0.f) return 0.f;

    // If the value is (very nearly) an integer in the MIDI range, treat it as MIDI.
    float vr = roundf(v);
    if (fabsf(v - vr) < 1e-6f && vr >= 0.f && vr <= 127.f){
        return jb_clamp(vr / 127.f, 0.f, 1.f);
    }

    // Otherwise, interpret values >1 as MIDI, <=1 as already-normalized.
    if (v > 1.f){
        return jb_clamp(v / 127.f, 0.f, 1.f);
    }
    return jb_clamp(v, 0.f, 1.f);
}

// RNG (xorshift64*) — per-voice per-ear, to keep stereo decorrelated
typedef struct { unsigned long long s; } jb_exc_rng64_t;
static inline void jb_exc_rng64_seed(jb_exc_rng64_t *r, unsigned long long seed){
    if(!seed) seed = 0x9E3779B97F4A7C15ull;
    r->s = seed;
}
static inline unsigned long long jb_exc_rng64_u64(jb_exc_rng64_t *r){
    unsigned long long x = r->s;
    x ^= x >> 12;
    x ^= x << 25;
    x ^= x >> 27;
    r->s = x;
    return x * 2685821657736338717ull;
}
static inline float jb_exc_rng64_uni(jb_exc_rng64_t *r){
    return (float)((jb_exc_rng64_u64(r) >> 40) * (1.0/16777216.0)); // 24-bit frac
}
static inline float jb_exc_noise_tpdf(jb_exc_rng64_t *r){
    return (jb_exc_rng64_uni(r) - jb_exc_rng64_uni(r)); // ~[-1,1]
}

// 1-pole filters
typedef struct { float a, y1, x1; } jb_exc_hp1_t;
typedef struct { float a, y1; } jb_exc_lp1_t;

static inline void jb_exc_hp1_set(jb_exc_hp1_t *f, float sr, float fc){
    if (fc <= 0.f){ f->a = -1.f; f->y1 = 0.f; f->x1 = 0.f; return; }
    float RC = 1.f / (2.f * (float)M_PI * fc);
    float dt = 1.f / sr;
    float alpha = RC / (RC + dt);
    if (alpha < 0.f) alpha = 0.f; else if (alpha > 1.f) alpha = 1.f;
    f->a = alpha;
}
static inline float jb_exc_hp1_run(jb_exc_hp1_t *f, float x){
    if (f->a < 0.f) return x;
    float y = f->a * (f->y1 + x - f->x1);
    f->y1 = y; f->x1 = x;
    return y;
}

static inline void jb_exc_lp1_set(jb_exc_lp1_t *f, float sr, float fc){
    if (fc <= 0.f){ f->a = 0.f; f->y1 = 0.f; return; }
    float a = expf(-2.f*(float)M_PI*fc/sr);
    f->a = a;
}
static inline float jb_exc_lp1_run(jb_exc_lp1_t *f, float x){
    float y = (1.f - f->a) * x + f->a * f->y1;
    f->y1 = y;
    return y;
}

// ADSR (per voice)
typedef enum { JB_EXC_ENV_IDLE=0, JB_EXC_ENV_ATTACK, JB_EXC_ENV_DECAY, JB_EXC_ENV_SUSTAIN, JB_EXC_ENV_RELEASE } jb_exc_env_stage;

typedef struct {
    jb_exc_env_stage stage;
    float env, sustain;
    int   a_n, d_n, r_n;
    int   a_i, d_i, r_i;
    float curveA, curveD, curveR; // -1..+1 per stage
    float kA, kD, kR;
    float release_start;
} jb_exc_adsr_t;

static inline float jb_exc_shape01(float u, float amt, float K){
    if (u <= 0.f) return 0.f;
    if (u >= 1.f) return 1.f;
    float a = jb_clamp(amt, -1.f, 1.f);
    if (a == 0.f) return u;
    float gamma = 1.f + fabsf(a) * K;
    if (a < 0.f) return 1.f - powf(1.f - u, gamma);
    return powf(u, gamma);
}
static inline float jb_exc_K_from_ms(float ms){
    float x = (ms < 0.f) ? 0.f : ms;
    return 4.f + 2.f*logf(1.f + 0.01f*x);
}
static inline int jb_exc_ms_to_samples(float sr, float ms){
    if (ms <= 0.f) return 0;
    float s = 0.001f * ms * sr;
    int n = (int)floorf(s + 0.5f);
    if (n < 1) n = 1;
    return n;
}
static inline void jb_exc_adsr_set_times(jb_exc_adsr_t *e, float sr, float a_ms, float d_ms, float s, float r_ms){
    e->a_n = jb_exc_ms_to_samples(sr, a_ms);
    e->d_n = jb_exc_ms_to_samples(sr, d_ms);
    e->r_n = (r_ms > 0.f) ? jb_exc_ms_to_samples(sr, r_ms) : 16;
    e->sustain = jb_clamp(s, 0.f, 1.f);
    e->kA = jb_exc_K_from_ms(a_ms);
    e->kD = jb_exc_K_from_ms(d_ms);
    e->kR = jb_exc_K_from_ms(r_ms);
}
static inline float jb_exc_adsr_next(jb_exc_adsr_t *e){
    switch(e->stage){
        case JB_EXC_ENV_IDLE: return 0.f;

        case JB_EXC_ENV_ATTACK:{
            if (e->a_n <= 0){ e->env = 1.f; e->stage = JB_EXC_ENV_DECAY; return e->env; }
            float u = (++e->a_i >= e->a_n) ? 1.f : (e->a_i / (float)e->a_n);
            float s = jb_exc_shape01(u, e->curveA, e->kA);
            e->env = s;
            if (e->a_i >= e->a_n){ e->stage = JB_EXC_ENV_DECAY; e->d_i = 0; }
        } break;

        case JB_EXC_ENV_DECAY:{
            if (e->d_n <= 0){ e->env = e->sustain; e->stage = JB_EXC_ENV_SUSTAIN; return e->env; }
            float u = (++e->d_i >= e->d_n) ? 1.f : (e->d_i / (float)e->d_n);
            float s = jb_exc_shape01(u, e->curveD, e->kD);
            e->env = e->sustain + (1.f - e->sustain) * (1.f - s);
            if (e->d_i >= e->d_n){ e->stage = JB_EXC_ENV_SUSTAIN; }
        } break;

        case JB_EXC_ENV_SUSTAIN:
            e->env = e->sustain;
            break;

        case JB_EXC_ENV_RELEASE:{
            if (e->r_n <= 0){ e->env = 0.f; e->stage = JB_EXC_ENV_IDLE; return 0.f; }
            float u = (++e->r_i >= e->r_n) ? 1.f : (e->r_i / (float)e->r_n);
            float s = jb_exc_shape01(u, e->curveR, e->kR);
            e->env = e->release_start * (1.f - s);
            if (e->r_i >= e->r_n){ e->env = 0.f; e->stage = JB_EXC_ENV_IDLE; }
        } break;
    }
    return e->env;
}
static inline void jb_exc_adsr_note_on(jb_exc_adsr_t *e){
    e->stage = JB_EXC_ENV_ATTACK;
    e->a_i = e->d_i = e->r_i = 0;
}
static inline void jb_exc_adsr_note_off(jb_exc_adsr_t *e){
    if (e->stage != JB_EXC_ENV_IDLE){
        e->release_start = e->env;
        e->stage = JB_EXC_ENV_RELEASE;
        e->r_i = 0;
    }
}

// Pulse generator (strike)
typedef struct {
    int   samples_left;
    float A, a1, a2, k2;
    float n;
} jb_exc_pulse_t;

// NOTE: Velocity scaling is applied uniformly in jb_exc_process_sample()
// so the impulse and noise branches share the same per-voice velocity->loudness law.
static inline void jb_exc_pulse_trigger(jb_exc_pulse_t *p){
    // Mathematically-correct impulse (delta): one sample with unit amplitude.
    // Any overall strike strength is applied later via per-voice velocity and gain scaling.
    p->A = 1.0f;
    p->samples_left = 1;
    p->n = 0.f;
}
static inline float jb_exc_pulse_next(jb_exc_pulse_t *p){
    if (p->samples_left <= 0) return 0.f;
    p->samples_left--;
    return p->A;
}

// Per-voice exciter runtime state (stereo)
typedef struct {
    float vel_cur;
    float vel_on;
    float pitch; // reserved for future pitch-shaped excitation
    jb_exc_rng64_t rngL, rngR;

    // Noise branch:
    //   - lpL/lpR are used as the low-band extractor for the slope-eq (pivot crossover)
    //   - hpL/hpR are used as a gentle DC high-pass after diffusion
    jb_exc_hp1_t hpL, hpR;
    jb_exc_lp1_t lpL, lpR;

    // Impulse branch filters (shape is impulse-only)
    jb_exc_hp1_t hpImpL, hpImpR;
    jb_exc_lp1_t lpImpL, lpImpR;

    jb_exc_adsr_t env;
    jb_exc_pulse_t pulseL, pulseR;

    float gainL, gainR;

    // Per-voice overrides for velocity mapping (optional)
    float color_gL, color_gH, color_comp;   // noise timbre/color gains
    float imp_shape_v;                      // 0..1, <0 => use global
    float noise_timbre_v;                   // 0..1, <0 => use global
    float a_ms_v, d_ms_v, r_ms_v;           // ADSR time overrides (ms), <0 => use global

} jb_exc_voice_t;

static void jb_exc_voice_init(jb_exc_voice_t *v, float sr, unsigned long long seed_base){
    memset(v, 0, sizeof(*v));

    jb_exc_rng64_seed(&v->rngL, seed_base + 1ull);
    jb_exc_rng64_seed(&v->rngR, seed_base + 2ull);

    v->env.stage = JB_EXC_ENV_IDLE;
    v->env.env = 0.f;
    v->env.sustain = 0.5f;
    v->env.curveA = 0.f;
    v->env.curveD = 0.f;
    v->env.curveR = 0.f;
    v->env.kA = v->env.kD = v->env.kR = 8.f;

    // filters will be configured in STEP 2 (per block): impulse-shape + noise-color/slope
    jb_exc_hp1_set(&v->hpL, sr, 0.f);
    jb_exc_hp1_set(&v->hpR, sr, 0.f);
    jb_exc_lp1_set(&v->lpL, sr, 0.f);
    jb_exc_lp1_set(&v->lpR, sr, 0.f);
    // impulse branch filters start identical but keep their own state
    jb_exc_hp1_set(&v->hpImpL, sr, 0.f);
    jb_exc_hp1_set(&v->hpImpR, sr, 0.f);
    jb_exc_lp1_set(&v->lpImpL, sr, 0.f);
    jb_exc_lp1_set(&v->lpImpR, sr, 0.f);

    v->gainL = 1.f;
    v->gainR = 1.f;

    // velocity-mapping overrides disabled by default
    v->color_gL = v->color_gH = 1.f;
    v->color_comp = 1.f;
    v->imp_shape_v = -1.f;
    v->noise_timbre_v = -1.f;
    v->a_ms_v = -1.f;
    v->d_ms_v = -1.f;
    v->r_ms_v = -1.f;

    // branch matching state
    // NOTE: old exciter noise diffusion (all-pass cascade) removed.
}

// STEP 2 helpers (runtime reset)
static inline void jb_exc_voice_reset_runtime(jb_exc_voice_t *e){
    // Keep RNG seeds/states (stereo decorrelation persists), but clear all time-varying state.
    e->vel_cur = 0.f;
    e->vel_on  = 0.f;
    e->pitch   = 0.f;

    // clear per-note overrides
    e->imp_shape_v = -1.f;
    e->noise_timbre_v = -1.f;
    e->a_ms_v = -1.f;
    e->d_ms_v = -1.f;
    e->r_ms_v = -1.f;

    // envelope
    e->env.stage = JB_EXC_ENV_IDLE;
    e->env.env = 0.f;
    e->env.a_i = e->env.d_i = e->env.r_i = 0;
    e->env.release_start = 0.f;

    // pulses
    e->pulseL.samples_left = 0; e->pulseL.n = 0.f;
    e->pulseR.samples_left = 0; e->pulseR.n = 0.f;

    // branch matching state


    // filter memories
    e->hpL.y1 = e->hpL.x1 = 0.f;
    e->hpR.y1 = e->hpR.x1 = 0.f;
    e->lpL.y1 = 0.f;
    e->lpR.y1 = 0.f;

    e->hpImpL.y1 = e->hpImpL.x1 = 0.f;
    e->hpImpR.y1 = e->hpImpR.x1 = 0.f;
    e->lpImpL.y1 = 0.f;
    e->lpImpR.y1 = 0.f;

    e->gainL = 1.f;
    e->gainR = 1.f;
}

static int jb_is_near_integer(float x, float eps){ float n=roundf(x); return fabsf(x-n)<=eps; }
static inline float jb_midi_to_hz(float n){ return 440.f * powf(2.f, (n-69.f)/12.f); }

typedef enum { DENSITY_PIVOT=0, DENSITY_INDIV=1 } jb_density_mode;

typedef struct {
    // base params (shared template per mode)
    float base_ratio, base_decay_ms, base_gain;
    int   active;
    int   keytrack; // 1 = track f0 (ratio), 0 = absolute Hz

    // signatures (random)
    float disp_signature;
    float micro_sig;
} jb_mode_base_t;

typedef struct {

    // runtime per-mode
    float ratio_now, decay_ms_now, gain_nowL, gain_nowR;
    float t60_s;

    // per-ear per-hit randomizations
    float md_hit_offsetL, md_hit_offsetR;   // micro detune offsets

    // ZDF SVF resonators (bandpass) per channel
    jb_svf_t svfL;
    jb_svf_t svfR;

    // drive/hit
    float driveL, driveR;
    int   hit_gateL, hit_coolL, hit_gateR, hit_coolR;

    // y_pre_last is used for hit-detection / safety meters (pre-pan, pre-master)
    float y_pre_lastL, y_pre_lastR;

    int   nyq_kill;
} jb_mode_rt_t;


typedef enum { V_IDLE=0, V_HELD=1, V_RELEASE=2 } jb_vstate;

typedef struct {
    jb_vstate state;
    float f0, vel, energy;
    // Voice-steal smoothing: if a new note arrives while this voice is active,
    // we do a tiny fade-out (release) then start the pending note.
    int   pending_note;
    int   steal_countdown;
    float pending_f0;
    float pending_vel;
    // Internal exciter runtime state (per voice, stereo)
    jb_exc_voice_t exc;
    // Per-voice LFO one-shot runtime (used when lfo_mode == 2)
    float   lfo_phase_state[JB_N_LFO];
    float   lfo_val[JB_N_LFO];
    float   lfo_snh[JB_N_LFO];
    uint8_t lfo_oneshot_done[JB_N_LFO];
    // Dedicated modulation ADSR (independent from exciter ADSR)

    // projected behavior (per voice) — BANK 1
    float pitch_x;
    float brightness_v;
    float decay_pitch_mul;
    float decay_vel_mul;
    float stiffness_add;

    // projected behavior — BANK 2
    float brightness_v2;
    float decay_pitch_mul2;
    float decay_vel_mul2;
    float stiffness_add2;

    // sympathetic multipliers — BANK 1
    // sympathetic multipliers — BANK 2
    // dispersion morph targets — BANK 1
    float disp_offset[JB_MAX_MODES];
    float disp_target[JB_MAX_MODES];

    // dispersion morph targets — BANK 2
    float disp_offset2[JB_MAX_MODES];
    float disp_target2[JB_MAX_MODES];

    // release envelopes (per-voice) — BANK 1/2
    float rel_env;
    float rel_env2;

    // Velocity-mapping per-note overrides (sentinels: <0 => use global/default)
    float velmap_pos[2];           // 0..1
    float velmap_pickup[2];        // 0..1
    float velmap_master_add[2];    // additive to per-voice bank gain (0..1 clamp later)
    float velmap_brightness_add[2];// additive to brightness (-1..+1 clamp later)
    float velmap_bell_zeta[2][JB_N_DAMPERS];   // override zeta_p (absolute)
    uint8_t velmap_bell_zeta_on[2][JB_N_DAMPERS];

    // Feedback loop state (per voice, per bank, 1-sample delayed)
    // Each bank ONLY feeds back its own output.
// Feedback delay + dispersion (per voice, per bank)

    // runtime per-mode — BANK 1/2
    jb_mode_rt_t m[JB_MAX_MODES];
    jb_mode_rt_t m2[JB_MAX_MODES];
} jb_voice_t;

// ---------- the object ----------
static t_class *juicy_bank_tilde_class;
static t_class *jb_tgtproxy_class;

/* -------------------------------------------------------------------------
   Cached Pd symbols (avoid gensym() in the audio thread)
   NOTE: gensym() touches Pd's symbol table; keep it out of perform().
   ------------------------------------------------------------------------- */
static t_symbol *jb_sym_master_1     = NULL;
static t_symbol *jb_sym_master_2     = NULL;
static t_symbol *jb_sym_lfo2_amount  = NULL;
static t_symbol *jb_sym_noise_timbre = NULL;
static t_symbol *jb_sym_imp_shape    = NULL;


// Proxy to accept ANY message on target-selection inlets (so message boxes like 'damper_1' work)
typedef struct _juicy_bank_tilde t_juicy_bank_tilde; // forward
typedef struct _jb_tgtproxy{
    t_pd p_pd;
    t_juicy_bank_tilde *owner;
    int lane; // 0=LFO1, 1=LFO2, 2=ADSR, 3=MIDI
} jb_tgtproxy;

// Proxy to accept ANY message on preset command inlet (so message boxes like 'FORWARD' work)
static t_class *jb_presetproxy_class;
typedef struct _jb_presetproxy{
    t_pd p_pd;
    t_juicy_bank_tilde *owner;
} jb_presetproxy;

static void jb_presetproxy_symbol(jb_presetproxy *p, t_symbol *s);
static void jb_presetproxy_anything(jb_presetproxy *p, t_symbol *s, int argc, t_atom *argv);
static void juicy_bank_tilde_preset_cmd(t_juicy_bank_tilde *x, t_symbol *s);



typedef struct _juicy_bank_tilde {

    // Pd object header (required for inlet_new/outlet_new, class registration, etc.)
    t_object x_obj;

    // Cached sample-rate (set in dsp(), used throughout DSP helpers)
    float sr;

    int n_modes;
    int active_modes;              // number of currently active partials (0..n_modes)
    int n_modes2;
    int active_modes2;
    // Bank editing focus (1-based UI: 1=bank1, 2=bank2)
    int   edit_bank;              // 0..1
    int   edit_damper;            // 0..JB_N_DAMPERS-1 (selected Type-4 damper to edit)
    float bank_master[2];          // per-bank master (0..1)
    int   bank_semitone[2];        // per-bank semitone transpose (-12..+12)
    int   bank_octave[2];          // per-bank octave (-2..+2, snapped)
    float bank_tune_cents[2];     // per-bank cents detune (-100..+100)
    // Individual/global inlets
    t_inlet *in_partials;          // message inlet for 'partials' (float 0..n_modes)
    t_inlet *in_master;            // per-bank master (selected bank)
    t_inlet *in_octave;      // per-bank octave (-2..+2)
    t_inlet *in_semitone;          // per-bank semitone transpose (selected bank)
    t_inlet *in_tune;              // per-bank cents detune (selected bank)
    t_inlet *in_bank;              // bank selector (1 or 2)
    // SPACE (global room)
    t_inlet *in_space_size;
    t_inlet *in_space_decay;
    t_inlet *in_space_diffusion;
    t_inlet *in_space_damping;
    t_inlet *in_space_wetdry;
    t_outlet *out_index;           // float outlet reporting current selected partial (1-based)
    jb_mode_base_t base[JB_MAX_MODES];
    jb_mode_base_t base2[JB_MAX_MODES];

    // BODY globals
    float release_amt;
    float release_amt2;

    
    // --- STRETCH (message-only, -1..1; internal musical scaling) ---
    float stretch;
    float stretch2;

    float warp; // -1..+1 bias for stretch distribution
    float warp2;
float brightness;

    // --- TYPE-4 CAUGHEY damping (Type-4 bell, stackable 3x per-bank) ---
    // One bell basis B(omega) centered at omega_p:
    //   B(omega) = (2 + n_pm) / ( (omega/omega_p)^(-n_pl) + n_pm + (omega/omega_p)^(n_pr) )
    //   zeta_k(omega) = zeta_p[k] * B_k(omega)
    // Stacking (faithful): zeta_total(omega) = sum_k zeta_k(omega)
    float bell_peak_hz[2][JB_N_DAMPERS];     // per-bank, per-damper peak frequency (Hz)
    float bell_peak_zeta[2][JB_N_DAMPERS];   // per-bank, per-damper peak damping ratio at the peak (zeta_p)
    float bell_peak_zeta_param[2][JB_N_DAMPERS]; // normalized 0..1 when set via bell_zeta inlet; <0 means 'off'
    float bell_npl[2][JB_N_DAMPERS];         // per-bank, per-damper left power  (n_pl > 0)
    float bell_npr[2][JB_N_DAMPERS];         // per-bank, per-damper right power (n_pr > 0)
    float bell_npm[2][JB_N_DAMPERS];         // per-bank, per-damper model parameter (n_pm > -2)

    float brightness2;

float density_amt; jb_density_mode density_mode;
    float density_amt2; jb_density_mode density_mode2;
    float dispersion, dispersion_last;
    float dispersion2, dispersion_last2;
    float odd_skew;
    float even_skew;
    float odd_skew2;
    float even_skew2;
    float collision_amt;
    float collision_amt2;

    // realism/misc
    float micro_detune;     // base for micro detune
    float micro_detune2;     // base for micro detune (bank2)
    float basef0_ref;

    // BEHAVIOR depths

    // voices
    int   max_voices;
    int   total_voices; // voices actually processed
    jb_voice_t v[JB_MAX_VOICES];

    // current edit index for Individual setters
    int edit_idx;

    int edit_idx2;

    // RNG
    jb_rng_t rng;

    // DC HP
    float hp_a, hpL_x1, hpL_y1, hpR_x1, hpR_y1;

    // SPACE parameters (0..1)
    float space_size;
    float space_decay;
    float space_diffusion;
    float space_damping;
    float space_wetdry;

    // SPACE state (global)
    float space_comb_buf[JB_SPACE_NCOMB][JB_SPACE_MAX_DELAY];
    int   space_comb_w[JB_SPACE_NCOMB];
    float space_comb_lp[JB_SPACE_NCOMB];

    float space_ap_buf[JB_SPACE_NAP][JB_SPACE_AP_MAX];
    int   space_ap_w[JB_SPACE_NAP];

    // IO
    // main stereo exciter inputs
    // per-voice exciter inputs (optional)

    t_outlet *outL, *outR;

    // INLET pointers
    // Behavior (reduced)
    
        t_inlet *in_release;
// Body controls (damping, brightness, density, dispersion, anisotropy)
    t_inlet *in_bell_peak_hz, *in_bell_peak_zeta, *in_bell_npl, *in_bell_npr, *in_bell_npm, *in_damper_sel, *in_brightness, *in_density, *in_stretch, *in_warp, *in_dispersion, *in_odd_skew, *in_even_skew, *in_collision;
    // Individual
    t_inlet *in_index, *in_ratio, *in_gain, *in_decay;
        // --- Spatial coupling (node/antinode; gain-level only) ---
    // Spatial excitation & pickup positions (1D, Elements-style)
    // Per-mode gain weight:
    //   w_i = sin(pi * fi * position) * sin(pi * fi * pickup)
    // where fi is the mode frequency ratio (mode_hz / f0). Works for inharmonic ratios too.
    // RMS normalization keeps loudness stable as position/pickup move:
    //   w_i_norm = w_i / sqrt(sum_i w_i^2)
    float excite_pos;   // 0..1 (excitation position)
    float pickup_pos;   // 0..1 (pickup/mic position)

    // Bank 2 (independent positions)
    float excite_pos2;  // 0..1 (excitation position, bank2)
    float pickup_pos2;  // 0..1 (pickup position, bank2)

    // Odd vs Even emphasis (-1..+1). Applied as an index-based gain mask.
    float odd_even_bias;  // bank1
    float odd_even_bias2; // bank2

    // inlet pointers for position controls
    t_inlet *in_position;
    t_inlet *in_pickup;

        t_inlet *in_odd_even; // odd vs even emphasis
// --- LFO globals (for modulation matrix UI) ---
    // lfo_shape / lfo_rate / lfo_phase always reflect the *currently selected* LFO,
    // as chosen by lfo_index (1 or 2). Per-LFO values live in the arrays below.
    float lfo_shape;   // 1..4 (1=saw,2=square,3=sine,4=SH)
    float lfo_rate;    // 1..20 Hz
    float lfo_phase;   // 0..1
    float lfo_mode;    // 1..2 (1=free, 2=one-shot)
    float lfo_index;   // 1 or 2 (selects which LFO)

    // per-LFO parameter storage
    float lfo_shape_v[JB_N_LFO];
    float lfo_rate_v[JB_N_LFO];
    float lfo_phase_v[JB_N_LFO];
    float lfo_mode_v[JB_N_LFO];

    // per-LFO runtime state (phase in cycles, current output, and S&H memory)
    float lfo_phase_state[JB_N_LFO];
    float lfo_val[JB_N_LFO];
    float lfo_snh[JB_N_LFO];

    // modulation matrix [modsource][target] amounts, -1..+1
    float mod_matrix[JB_N_MODSRC][JB_N_MODTGT];
    float mod_matrix2[JB_N_MODSRC][JB_N_MODTGT];

    // --- NEW MOD LANES (matrix replacement scaffolding; target wiring comes next) ---
    // Targets are stored as symbols; the special symbol "none" disables that lane.
    // Uniqueness rule: if a target is already owned by another lane, new assignment is ignored.
    t_symbol *lfo_target[JB_N_LFO];
    // Velocity mapping lane (velocity -> selected target per note)
    float     velmap_amount;         // -1..+1
    t_symbol *velmap_target;         // symbol selector

        uint8_t   velmap_on[JB_VELMAP_N_TARGETS];         // toggle map of enabled velocity-mapping targets (see enum jb_velmap_idx)
    jb_tgtproxy *tgtproxy_velmap;
    t_inlet  *in_velmap_amount;
    t_inlet  *in_velmap_target;

// --- PRESET SYSTEM (memory-only) ---
t_inlet  *in_preset_cmd;      // ANY message inlet: INIT/SAVE/FORWARD/BACKWARD (via jb_presetproxy)
t_inlet  *in_preset_char;     // float inlet: 1..64 character selector (space + A..Z + a..z + 0..9 + extra)
t_outlet *out_preset;         // symbol outlet for preset UI feedback
t_outlet *out_preset_f;       // float outlet: emits preset_mode (0/1/2) for easy [print]
jb_presetproxy *presetproxy;  // proxy that receives preset_cmd messages

jb_preset_t presets[JB_PRESET_SLOTS];
int preset_mode;              // jb_preset_mode_t
int preset_cursor;            // 0..JB_PRESET_NAME_MAX-1 (naming mode)
int preset_slot_sel;          // 0..JB_PRESET_SLOTS-1 (slot mode)
char preset_edit_name[JB_PRESET_NAME_MAX + 1];
   // LFO1/LFO2 targets
    jb_tgtproxy *tgtproxy_lfo1;
    jb_tgtproxy *tgtproxy_lfo2;
    float     lfo_amt_v[JB_N_LFO];    // LFO1/LFO2 amounts (-1..+1)
    float     lfo_amt_eff[JB_N_LFO];  // effective amounts (after LFO1->LFO2 amount mod)
    float     lfo_amount;             // UI mirror for currently selected LFO amount (via lfo_index)

    // Dedicated modulation ADSR (independent of exciter ADSR)

    // MIDI lane

    // --- INTERNAL EXCITER params (shared, Fusion STEP 1) ---
    float exc_fader;
    float exc_attack_ms, exc_attack_curve;
    float exc_decay_ms,  exc_decay_curve;
    float exc_sustain;
    float exc_release_ms, exc_release_curve;
    // exc_density: repurposed -> Pressure (AGC target level for feedback loop, 0..1 -> 0..0.98)
    float exc_density;
    // exc_imp_shape: impulse-only shape (0..1)
    float exc_imp_shape;
    // exc_shape: repurposed -> Noise Color (0..1; red..white..violet)
    float exc_shape;
    // Feedback loop extra shaping (Prism-style)
    // per-block computed (shared)
    float exc_noise_color_gL, exc_noise_color_gH, exc_noise_color_comp;

    // --- FEEDBACK AGC (per-bank, per-voice) ---
    // Attack/Release are 0..1 UI parameters mapped to time constants (seconds).

    float noise_scale;   // factory trim (linear amp)
    float impulse_scale; // factory trim (linear amp)
    // --- INTERNAL EXCITER inlets (created after keytrack, before LFO) ---
    t_inlet *in_exc_fader;
    t_inlet *in_exc_attack;
    t_inlet *in_exc_attack_curve;
    t_inlet *in_exc_decay;
    t_inlet *in_exc_decay_curve;
    t_inlet *in_exc_sustain;
    t_inlet *in_exc_release;
    t_inlet *in_exc_release_curve;
    t_inlet *in_exc_imp_shape;
    t_inlet *in_exc_shape;

    // AGC inlets (placed after timbre/color, before LFO index)

    // --- MOD SECTION inlets (targets/amounts; actual wiring added next step) ---
    t_inlet *in_lfo_index;
    t_inlet *in_lfo_shape;
    t_inlet *in_lfo_rate;
    t_inlet *in_lfo_phase;
    t_inlet *in_lfo_mode;
    t_inlet *in_lfo_amount;
    t_inlet *in_lfo1_target;
    t_inlet *in_lfo2_target;


    // Offline render buffer (testing/regression)
    float *render_bufL;
    float *render_bufR;
    int   render_len;   // samples per channel
    int   render_sr;    // sample rate used for the render

// --- CHECKPOINT (bake) revert buffer ---
} t_juicy_bank_tilde;

// ---------- LFO runtime update (per block) ----------
// Updates both LFOs for this block. Outputs live in x->lfo_val[0..JB_N_LFO-1],
// normalised to -1..+1 for all shapes.
static void jb_update_lfos_block(t_juicy_bank_tilde *x, int n){
    if (x->sr <= 0.f || n <= 0){
        for (int li = 0; li < JB_N_LFO; ++li){
        int mode = (int)floorf(x->lfo_mode_v[li] + 0.5f);
        if (mode == 2){
            // one-shot LFO is computed per-voice (see jb_update_lfos_oneshot_voice_block)
            x->lfo_val[li] = 0.f;
            continue;
        }
            x->lfo_val[li] = 0.f;
        }
        return;
    }

    const float inv_sr = 1.f / x->sr;

    for (int li = 0; li < JB_N_LFO; ++li){
        float rate  = jb_clamp(x->lfo_rate_v[li], 0.f, 20.f);   // Hz
        if (li == 1) {
            const t_symbol *tgt = x->lfo_target[0];
            if (tgt == gensym("lfo2_rate")) {
                const float lfo1_amt = jb_clamp(x->lfo_amt_v[0], -1.f, 1.f);
                const float lfo1_out = x->lfo_val[0] * lfo1_amt;
                rate = jb_clamp(rate + lfo1_out, 0.f, 20.f);
            }
        }
        float shape_f = x->lfo_shape_v[li];
        float phase_off = x->lfo_phase_v[li];

        if (rate <= 0.f){
            x->lfo_val[li] = 0.f;
            continue;
        }

        // advance phase in *cycles* (0..1)
        float phase = x->lfo_phase_state[li];
        const float dcycles = rate * ((float)n * inv_sr);
        float prev_phase = phase;
        phase += dcycles;
        phase -= floorf(phase); // wrap to 0..1

        // apply user phase offset
        float ph = phase + phase_off;
        ph -= floorf(ph);

        int shape = (int)floorf(shape_f + 0.5f);
        if (shape < 1) shape = 1;
        if (shape > 5) shape = 5;

        float val = 0.f;
        if (shape == 1){
            // saw (up): -1..+1
            val = 2.f * ph - 1.f;
        } else if (shape == 5){
            // saw (down): +1..-1
            val = 1.f - 2.f * ph;
        } else if (shape == 2){
            // square
            val = (ph < 0.5f) ? 1.f : -1.f;
        } else if (shape == 3){
            // sine
            val = sinf(2.f * M_PI * ph);
        } else {
            // shape == 4 : sample & hold noise
            // generate a new random value whenever the phase wraps around
            if (phase < prev_phase){
                x->lfo_snh[li] = jb_rng_bi(&x->rng);
            }
            val = x->lfo_snh[li];
        }

        x->lfo_phase_state[li] = phase;
        x->lfo_val[li] = val;
    }

}

static inline float jb_lfo_value_for_voice(const t_juicy_bank_tilde *x, const jb_voice_t *v, int li){
    // li: 0=LFO1, 1=LFO2
    int m = (int)floorf((li >= 0 && li < JB_N_LFO ? x->lfo_mode_v[li] : 1.f) + 0.5f);
    if (m == 2){
        return jb_clamp(v->lfo_val[li], -1.f, 1.f);
    }
    return jb_clamp(x->lfo_val[li], -1.f, 1.f);
}

// Update per-voice one-shot LFO state for this block (only when lfo_mode == 2).
// Behavior:
//   • One-shot shapes (saw/square/sine) run exactly one cycle, then hold the final value.
//   • One-shot S&H outputs a single random value per note-on and holds it.
static inline void jb_update_lfos_oneshot_voice_block(t_juicy_bank_tilde *x, jb_voice_t *v, int n){
    if (x->sr <= 0.f || n <= 0) return;
    const float inv_sr = 1.f / x->sr;

    for (int li = 0; li < JB_N_LFO; ++li){
        int mode = (int)floorf(x->lfo_mode_v[li] + 0.5f);
        if (mode != 2) continue;

        // S&H one-shot is handled at note-on (one random value, held).
        int shape = (int)floorf(x->lfo_shape_v[li] + 0.5f);
        if (shape < 1) shape = 1;
        if (shape > 5) shape = 5;
        if (shape == 4){
            continue;
        }

        if (v->lfo_oneshot_done[li]) continue;

        float rate = jb_clamp(x->lfo_rate_v[li], 0.f, 20.f);
        if (rate <= 0.f){
            v->lfo_val[li] = 0.f;
            v->lfo_oneshot_done[li] = 1;
            continue;
        }

        float phase = v->lfo_phase_state[li];
        phase += rate * ((float)n * inv_sr);

        if (phase >= 1.f){
            phase = 1.f;
            v->lfo_oneshot_done[li] = 1;
        }

        float ph = phase + x->lfo_phase_v[li];
        // In one-shot mode we clamp (no wrap) so the "one cycle" stays one cycle.
        ph = jb_clamp(ph, 0.f, 1.f);

        float val = 0.f;
        if (shape == 1){
            val = 2.f * ph - 1.f;
        } else if (shape == 5){
            val = 1.f - 2.f * ph;
        } else if (shape == 2){
            val = (ph < 0.5f) ? 1.f : -1.f;
        } else { // shape == 3 (sine)
            val = sinf(2.f * M_PI * ph);
        }

        v->lfo_phase_state[li] = phase;
        v->lfo_val[li] = val;
    }
}

 // ---------- modulation-source normalisation helper ----------

// Returns a normalised value for each modulation source:
//   0: velocity  -> 0..1
//   1: pitch     -> -1..+1 (approx. +/- two octaves around basef0_ref)
//   2: adsr      -> 0..1 (envelope from exciter)
//   3: lfo1      -> -1..+1
//   4: lfo2      -> -1..+1
static float jb_mod_source_value(const t_juicy_bank_tilde *x,
                                 const jb_voice_t *v,
                                 int src_idx)
{
    switch (src_idx){
    case 0: // velocity
        return jb_clamp(v->vel, 0.f, 1.f);

    case 1: // pitch: normalise semitone offset from basef0_ref (clamped)
    {
        float f0 = (v->f0 > 1e-6f) ? v->f0 : 1.f;
        float ratio = f0 / ((x->basef0_ref > 1e-6f) ? x->basef0_ref : 110.f);
        float semi  = 12.f * logf(ratio) / logf(2.f);
        // map ~[-48,+48] -> [0,1]
        float norm = (semi + 48.f) / 96.f;
        return jb_clamp(norm, 0.f, 1.f);
    }

    case 2: // lfo1 (-1..+1)
        return jb_lfo_value_for_voice(x, v, 0);

    case 3: // lfo2 (-1..+1)
        return jb_lfo_value_for_voice(x, v, 1);

    default:
        return 0.f;
    }
}


// ---------- INTERNAL EXCITER (Fusion STEP 2) — block update + per-sample render ----------

// Update all per-voice exciter parameters that depend on the shared inlets.
// Called once per DSP block.
static void jb_exc_update_block(t_juicy_bank_tilde *x){
    float sr = (x->sr > 0.f) ? x->sr : 48000.f;

    // Shared ADSR curves (times may be overridden per voice by velocity mapping)
    float aC = jb_clamp(x->exc_attack_curve,  -1.f, 1.f);
    float dC = jb_clamp(x->exc_decay_curve,   -1.f, 1.f);
    float rC = jb_clamp(x->exc_release_curve, -1.f, 1.f);

    // Shared base ADSR times
    float a_ms_base = x->exc_attack_ms;
    float d_ms_base = x->exc_decay_ms;
    float sus       = x->exc_sustain;
    float r_ms_base = x->exc_release_ms;

    // Shared noise pivot filters (slope-EQ split) — same for all voices
    float pivot_hz = JB_EXC_SLOPE_PIVOT_HZ;
    if (pivot_hz < 50.f) pivot_hz = 50.f;
    if (pivot_hz > 0.45f * sr) pivot_hz = 0.45f * sr;

    for(int i=0; i<x->total_voices; ++i){
        jb_exc_voice_t *e = &x->v[i].exc;

        // ----- Noise timbre/color (0..1) -----
        float color = (e->noise_timbre_v >= 0.f) ? jb_clamp(e->noise_timbre_v, 0.f, 1.f)
                                                 : jb_clamp(x->exc_shape, 0.f, 1.f);
        float slope_db_per_oct = -6.f + 12.f * color;
        float slope_db = slope_db_per_oct * JB_EXC_COLOR_OCT_SPAN;
        float gH = powf(10.f,  slope_db / 20.f);
        float gL = powf(10.f, -slope_db / 20.f);
        float comp = 1.f / sqrtf(0.5f * (gL*gL + gH*gH) + 1e-12f);

        e->color_gL = gL;
        e->color_gH = gH;
        e->color_comp = comp;

        // Noise filters (pivot LP + DC HP)
        jb_exc_lp1_set(&e->lpL, sr, pivot_hz);
        jb_exc_lp1_set(&e->lpR, sr, pivot_hz);
        jb_exc_hp1_set(&e->hpL, sr, 5.f);
        jb_exc_hp1_set(&e->hpR, sr, 5.f);

        // ----- Impulse shape (0..1) -----
        float s = (e->imp_shape_v >= 0.f) ? jb_clamp(e->imp_shape_v, 0.f, 1.f)
                                          : jb_clamp(x->exc_imp_shape, 0.f, 1.f);

        float lp_norm, hp_norm;
        if (s <= 0.5f){
            float t = s / 0.5f;
            lp_norm = t;
            hp_norm = 0.f;
        }else{
            float t = (s - 0.5f) / 0.5f;
            lp_norm = 1.f;
            hp_norm = t;
        }
        float lp_min = 200.f, lp_max = 0.48f * sr;
        float hp_min = 5.f,   hp_max = 8000.f;

        float lp_hz = jb_exc_expmap01(jb_clamp(lp_norm,0.f,1.f), lp_min, lp_max);
        float hp_hz = jb_exc_expmap01(jb_clamp(hp_norm,0.f,1.f), hp_min, hp_max);
        if (lp_hz < hp_hz + 50.f) lp_hz = hp_hz + 50.f;

        jb_exc_hp1_set(&e->hpImpL, sr, hp_hz);
        jb_exc_hp1_set(&e->hpImpR, sr, hp_hz);
        jb_exc_lp1_set(&e->lpImpL, sr, lp_hz);
        jb_exc_lp1_set(&e->lpImpR, sr, lp_hz);

        // ----- ADSR (times optionally overridden per voice) -----
        float a_ms = (e->a_ms_v >= 0.f) ? e->a_ms_v : a_ms_base;
        float d_ms = (e->d_ms_v >= 0.f) ? e->d_ms_v : d_ms_base;
        float r_ms = (e->r_ms_v >= 0.f) ? e->r_ms_v : r_ms_base;

        e->env.curveA = aC;
        e->env.curveD = dC;
        e->env.curveR = rC;
        jb_exc_adsr_set_times(&e->env, sr, a_ms, d_ms, sus, r_ms);
    }
}


// Update per-sample increments for the dedicated modulation ADSR.
// Called once per DSP block.
// (removed stray text line that broke compilation)
// Called once per DSP block.



// Fractional-delay Schroeder all-pass (linear interpolation read).
// Buffer stores w[n] = x[n] + g*y[n].
static inline float jb_exc_apf_run(float x, float g, float delay_samps, float *buf, int *w, int maxlen){
    if (g <= 0.f) return x;
    float d = jb_clamp(delay_samps, 1.f, (float)(maxlen - 2));
    int wi = *w;

    float r = (float)wi - d;
    while (r < 0.f) r += (float)maxlen;

    int i0 = (int)r;
    float frac = r - (float)i0;
    int i1 = i0 + 1;
    if (i1 >= maxlen) i1 -= maxlen;

    float wd = buf[i0] + (buf[i1] - buf[i0]) * frac;
    float y = -g * x + wd;
    float wnew = x + g * y;

    buf[wi] = wnew;
    wi++;
    if (wi >= maxlen) wi = 0;
    *w = wi;
    return y;
}



// 1-pole high-pass (stateful): y[n] = a*(y[n-1] + x[n] - x[n-1])
static inline float jb_hp1_run_a(float x, float a, float *x1, float *y1){
    float y = a * ((*y1) + x - (*x1));
    *x1 = x;
    *y1 = y;
    return y;
}

// Soft clip with ceiling=1.0 and a lower knee (threshold) where it begins to squash.
// threshold should be ~0.7..0.8; we default to 0.75 for organic saturation.
static inline float jb_softclip_thresh(float x, float threshold){
    float t = threshold;
    if (t < 0.05f) t = 0.05f;
    if (t > 0.95f) t = 0.95f;

    float ax = fabsf(x);
    if (ax <= t) return x;

    float s = (ax - t) / (1.f - t);           // 0..inf
    float y = t + (1.f - t) * tanhf(s);       // asymptote -> 1.0
    return copysignf(y, x);
}

// Render one exciter sample for one voice (stereo).
static inline void jb_exc_process_sample(const t_juicy_bank_tilde *x,
                                         jb_voice_t *v,
                                         float w_imp, float w_noise,
                                         float *outL, float *outR)
{
    jb_exc_voice_t *e = &v->exc;

    float env = jb_exc_adsr_next(&e->env);

    // fast silent path (env off + no active pulses)
    if (e->env.stage == JB_EXC_ENV_IDLE && e->pulseL.samples_left<=0 && e->pulseR.samples_left<=0){
        *outL = 0.f;
        *outR = 0.f;
        return;
    }

    // ---------- NOISE BRANCH ----------
    float nL = jb_exc_noise_tpdf(&e->rngL);
    float nR = jb_exc_noise_tpdf(&e->rngR);

    // Noise Color (slope EQ): split around pivot using 1-pole LP, then re-weight low/high.
    float lpL = jb_exc_lp1_run(&e->lpL, nL);
    float lpR = jb_exc_lp1_run(&e->lpR, nR);
    float hpL = nL - lpL;
    float hpR = nR - lpR;

    float colL = e->color_comp * (e->color_gL * lpL + e->color_gH * hpL);
    float colR = e->color_comp * (e->color_gL * lpR + e->color_gH * hpR);
    // DC protection
    colL = jb_exc_hp1_run(&e->hpL, colL);
    colR = jb_exc_hp1_run(&e->hpR, colR);

    float yL = colL * env * e->vel_on * e->gainL;
    float yR = colR * env * e->vel_on * e->gainR;

    // ---------- IMPULSE BRANCH (shape affects impulse only) ----------
    float pL = jb_exc_pulse_next(&e->pulseL);
    float pR = jb_exc_pulse_next(&e->pulseR);

    pL = jb_exc_lp1_run(&e->lpImpL, jb_exc_hp1_run(&e->hpImpL, pL));
    pR = jb_exc_lp1_run(&e->lpImpR, jb_exc_hp1_run(&e->hpImpR, pR));

    // Impulse is NOT governed by the exciter ADSR (noise is). We still scale by velocity and per-voice micro-variation.
    pL *= e->vel_on * e->gainL;
    pR *= e->vel_on * e->gainR;


    *outL = w_noise * (yL * x->noise_scale) + w_imp * (pL * x->impulse_scale);
    *outR = w_noise * (yR * x->noise_scale) + w_imp * (pR * x->impulse_scale);
}
// ---------

// ---------- helpers ----------
static float jb_bright_gain(float ratio_rel, float b){
    // Brightness slope (defines spectral slope):
    //  - b in [-1,1]
    //  - b = 0   : saw reference (gain ~ 1/ratio_rel)
    //  - b = +1  : flat spectrum (all modes ~ equal gain)
    //  - b = -1  : dark (gain ~ 1/ratio_rel^2)
    //
    // Implementation: gain = ratio_rel^(-alpha), alpha = 1 - b
    //  -> b=-1 => alpha=2 ; b=0 => alpha=1 ; b=+1 => alpha=0
    float bb = jb_clamp(b, -1.f, 1.f);
    float rr = jb_clamp(ratio_rel, 1.f, 1e6f);
    float alpha = 1.f - bb;
    return powf(rr, -alpha);
}
// ---------- density mapping ----------
// Only keytracked modes are spread by density; absolute-Hz modes keep base_ratio.
static void jb_apply_density(const t_juicy_bank_tilde *x, jb_voice_t *v){
    // New density mapping:
    //  • Only keytracked, active modes are affected; absolute-Hz modes keep base_ratio.
    //  • density_amt is interpreted in "harmonic gap units":
    //      0   -> gap = 1  (1x, 2x, 3x, ... : normal harmonic spacing)
    //      1   -> gap = 2  (1x, 3x, 5x, ... : every other harmonic)
    //      2   -> gap = 3  (1x, 4x, 7x, ... : skipping 2 harmonics, etc.)
    //     -1   -> gap = 0  (all keytracked modes collapse onto the fundamental)
    //  • Negative side is clamped to -1, positive side is unbounded.
    //
    // 1. Collect keytracked modes, keep absolute-Hz modes unchanged.
    float dens = x->density_amt;
    if (dens < -1.f)
        dens = -1.f;

    int idxs[JB_MAX_MODES];
    int count = 0;
    for (int i = 0; i < x->n_modes; ++i){
        if (x->base[i].active && x->base[i].keytrack){
            idxs[count++] = i;
        } else {
            // absolute-Hz or inactive modes: keep their base ratio
            v->m[i].ratio_now = x->base[i].base_ratio;
        }
    }
    if (count == 0)
        return;

    // 2. Sort keytracked modes by their base_ratio so we can apply an ordered gap.
    for (int k = 1; k < count; ++k){
        int id = idxs[k];
        int j  = k;
        while (j > 0 && x->base[idxs[j-1]].base_ratio > x->base[id].base_ratio){
            idxs[j] = idxs[j-1];
            --j;
        }
        idxs[j] = id;
    }

    // 3. Find the "fundamental" among keytracked modes: the one closest to ratio = 1.
    int   pivot_j = 0;
    float best    = 1e9f;
    for (int j = 0; j < count; ++j){
        int   id = idxs[j];
        float d  = fabsf(x->base[id].base_ratio - 1.f);
        if (d < best){
            best    = d;
            pivot_j = j;
        }
    }
    int pivot_id = idxs[pivot_j];
    float r_pivot = x->base[pivot_id].base_ratio;

    // 4. Map density to a harmonic gap.
    //    1 whole unit of density corresponds to 1 whole additional integer gap.
    //    gap = 1 + dens, clamped so it never goes negative.
    float gap = 1.f + dens;
    if (gap < 0.f)
        gap = 0.f;

    // 5. Write new ratios: equally spaced by "gap" around the pivot.
    for (int j = 0; j < count; ++j){
        int id   = idxs[j];
        int step = j - pivot_j;            // 0 at pivot (fundamental)
        float r  = r_pivot + gap * (float)step;
        if (r < 0.01f)
            r = 0.01f;
        v->m[id].ratio_now = r;
    }
}
// --- Fundamental lock: keep mode 0 at exact x1 if keytracked ---
static void jb_lock_fundamental_after_density(const t_juicy_bank_tilde *x, jb_voice_t *v){
    if (x->n_modes > 0 && x->base[0].keytrack){
        v->m[0].ratio_now = 1.f;
    }

}

// ---------- bank-aware helper accessors ----------
static inline int jb_bank_nmodes(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->n_modes2 : x->n_modes;
}
static inline int jb_bank_active_modes(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->active_modes2 : x->active_modes;
}
static inline const jb_mode_base_t *jb_bank_base(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->base2 : x->base;
}
static inline float jb_bank_dispersion(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->dispersion2 : x->dispersion;
}
static inline float *jb_bank_dispersion_last(t_juicy_bank_tilde *x, int bank){
    return bank ? &x->dispersion_last2 : &x->dispersion_last;
}
static inline float jb_bank_odd_skew(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->odd_skew2 : x->odd_skew;
}
static inline float jb_bank_even_skew(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->even_skew2 : x->even_skew;
}
static inline float jb_bank_collision_amt(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->collision_amt2 : x->collision_amt;
}
static inline float jb_bank_density_amt(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->density_amt2 : x->density_amt;
}
static inline float jb_bank_stretch_amt(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->stretch2 : x->stretch;
}
static inline float jb_bank_warp_amt(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->warp2 : x->warp;
}

// --- TYPE-4 Caughey damping (Type-4 bell, stackable 3x per-bank) ---
// We operate directly in modal coordinates:
//   zeta_k(omega) = zeta_p[k] * B_k(omega)
//   B_k(omega)    = (2 + n_pm[k]) / ( (omega/omega_p[k])^(-n_pl[k]) + n_pm[k] + (omega/omega_p[k])^(n_pr[k]) )
// Stacking (faithful):
//   zeta_total(omega) = sum_{k=0..JB_N_DAMPERS-1} zeta_k(omega)
// Then we convert zeta to T60 via: exp(-zeta*omega*T60)=1/1000.

static inline float jb_bank_bell_peak_hz(const t_juicy_bank_tilde *x, int bank, int damper){
    if (bank < 0) bank = 0; else if (bank > 1) bank = 1;
    if (damper < 0) damper = 0; else if (damper >= JB_N_DAMPERS) damper = JB_N_DAMPERS - 1;
    return x->bell_peak_hz[bank][damper];
}
static inline float jb_bank_bell_peak_zeta(const t_juicy_bank_tilde *x, int bank, int damper){
    if (bank < 0) bank = 0; else if (bank > 1) bank = 1;
    if (damper < 0) damper = 0; else if (damper >= JB_N_DAMPERS) damper = JB_N_DAMPERS - 1;
    return x->bell_peak_zeta[bank][damper];
}
static inline float jb_bank_bell_npl(const t_juicy_bank_tilde *x, int bank, int damper){
    if (bank < 0) bank = 0; else if (bank > 1) bank = 1;
    if (damper < 0) damper = 0; else if (damper >= JB_N_DAMPERS) damper = JB_N_DAMPERS - 1;
    return x->bell_npl[bank][damper];
}
static inline float jb_bank_bell_npr(const t_juicy_bank_tilde *x, int bank, int damper){
    if (bank < 0) bank = 0; else if (bank > 1) bank = 1;
    if (damper < 0) damper = 0; else if (damper >= JB_N_DAMPERS) damper = JB_N_DAMPERS - 1;
    return x->bell_npr[bank][damper];
}
static inline float jb_bank_bell_npm(const t_juicy_bank_tilde *x, int bank, int damper){
    if (bank < 0) bank = 0; else if (bank > 1) bank = 1;
    if (damper < 0) damper = 0; else if (damper >= JB_N_DAMPERS) damper = JB_N_DAMPERS - 1;
    return x->bell_npm[bank][damper];
}

static inline float jb_type4_basis(float omega, float omega_p, float npl, float npr, float npm){
    // Safety clamps (avoid NaNs and preserve the model constraints)
    if (!isfinite(omega)) omega = 0.f;
    if (omega < 0.f) omega = -omega;
    if (!isfinite(omega_p) || omega_p < 1e-6f) omega_p = 1e-6f;

    if (!isfinite(npl) || npl < 1e-4f) npl = 1e-4f;
    if (!isfinite(npr) || npr < 1e-4f) npr = 1e-4f;
    if (!isfinite(npm)) npm = 0.f;
    if (npm <= -1.99f) npm = -1.99f; // must be > -2

    float x = omega / omega_p;
    if (!isfinite(x) || x < 1e-12f) x = 1e-12f;

    // Use powf with safe exponents. Since x>0, this is fine.
    float left  = powf(x, -npl);
    float right = powf(x,  npr);

    float denom = left + npm + right;
    float numer = 2.f + npm;

    if (!isfinite(denom) || denom <= 1e-20f) return 0.f;
    float B = numer / denom;
    if (!isfinite(B) || B < 0.f) B = 0.f;
    return B;
}

static inline float jb_bell_zeta_eval(const t_juicy_bank_tilde *x, const jb_voice_t *v, int bank, float omega){
    // Parameters (already mapped/clamped by inlet handlers)
    // Stacked Type-4 bell damping: zeta_total(omega) = sum_k zeta_k(omega)
    float zsum = 0.f;

    // avoid peaks above Nyquist (if sr is known)
    float sr = x->sr;
    float nyq = (sr > 1.f) ? (0.5f * sr) : 0.f;

    for (int k = 0; k < JB_N_DAMPERS; ++k){
        float zeta_p = (v && v->velmap_bell_zeta_on[bank][k]) ? v->velmap_bell_zeta[bank][k]
                     : jb_bank_bell_peak_zeta(x, bank, k);
        if (!isfinite(zeta_p) || zeta_p <= 0.f) continue; // treat <=0 as "off"

        float peak_hz = jb_bank_bell_peak_hz(x, bank, k);
        float npl     = jb_bank_bell_npl(x, bank, k);
        float npr     = jb_bank_bell_npr(x, bank, k);
        float npm     = jb_bank_bell_npm(x, bank, k);

        // Convert peak frequency to omega_p (rad/sec)
        if (!isfinite(peak_hz) || peak_hz < 1.f) peak_hz = 1.f;
        if (nyq > 0.f && peak_hz > 0.95f * nyq) peak_hz = 0.95f * nyq;

        float omega_p = 2.f * (float)M_PI * peak_hz;

        float B = jb_type4_basis(omega, omega_p, npl, npr, npm);
        float z = zeta_p * B;

        if (!isfinite(z) || z < 0.f) z = 0.f;
        zsum += z;
    }

    if (!isfinite(zsum) || zsum < 0.f) zsum = 0.f;
    // Hard cap to keep it sane (zeta above ~0.2 dies basically instantly)
    if (zsum > 0.2f) zsum = 0.2f;
    return zsum;
}

static inline float jb_bank_brightness(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->brightness2 : x->brightness;
}

static inline float jb_bank_micro_detune(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->micro_detune2 : x->micro_detune;
}
static inline float jb_bank_release_amt(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->release_amt2 : x->release_amt;
}
static inline float jb_bank_excite_pos(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->excite_pos2 : x->excite_pos;
}
static inline float jb_bank_pickup_pos(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->pickup_pos2 : x->pickup_pos;
}

static inline float (*jb_bank_mod_matrix(t_juicy_bank_tilde *x, int bank))[JB_N_MODTGT]{
    return bank ? x->mod_matrix2 : x->mod_matrix;
}

// ---------- generic bank-safe transforms on ratio_now ----------
static void jb_apply_density_generic(int n_modes, const jb_mode_base_t *base, float density_amt, jb_mode_rt_t *m){
    float dens = density_amt;
    if (dens < -1.f) dens = -1.f;

    int idxs[JB_MAX_MODES];
    int count = 0;

    for (int i = 0; i < n_modes; ++i){
        if (base[i].active && base[i].keytrack){
            idxs[count++] = i;
        } else {
            m[i].ratio_now = base[i].base_ratio;
        }
    }
    if (count == 0) return;

    // sort by base ratio
    for (int k = 1; k < count; ++k){
        int id = idxs[k];
        int j  = k;
        while (j > 0 && base[idxs[j-1]].base_ratio > base[id].base_ratio){
            idxs[j] = idxs[j-1];
            --j;
        }
        idxs[j] = id;
    }

    // pivot nearest ratio=1
    int pivot_j = 0;
    float best = 1e9f;
    for (int j = 0; j < count; ++j){
        int id = idxs[j];
        float d = fabsf(base[id].base_ratio - 1.f);
        if (d < best){ best = d; pivot_j = j; }
    }
    int pivot_id = idxs[pivot_j];
    float r_pivot = base[pivot_id].base_ratio;

    float gap = 1.f + dens;
    if (gap < 0.f) gap = 0.f;

    for (int j = 0; j < count; ++j){
        int id = idxs[j];
        int step = j - pivot_j;
        float r = r_pivot + gap * (float)step;
        if (r < 0.01f) r = 0.01f;
        m[id].ratio_now = r;
    }
}

static void jb_lock_fundamental_generic(int n_modes, const jb_mode_base_t *base, jb_mode_rt_t *m){
    if (n_modes > 0 && base[0].keytrack){
        m[0].ratio_now = 1.f;
    }
}


static void jb_apply_stretch_generic(int n_modes, const jb_mode_base_t *base, float stretch, float warp, jb_mode_rt_t *m){
    // "Normal" stretch/warp:
    //  - Work in log-frequency (octaves): d = log2(ratio)
    //  - Stretch (s) expands (+) or contracts (-) partial spacing progressively toward the top.
    //  - Warp (w) shapes *where* the bend is concentrated (low vs high spectrum).
    //
    //  u = rank/(count-1) in [0..1] (rank = low->high in the modal sequence)
    //  p = 2^(w*2)  => w=-1..1 maps to p=0.25..4
    //  b(u) = u^p
    //  d' = d0 + (d-d0) * (1 + s*b(u))
    //
    //  This anchors the lowest mode (d0) and avoids any moving "pivot".
    float s = jb_clamp(stretch, -1.f, 1.f);
    float w = jb_clamp(warp,   -1.f, 1.f);
    if (s == 0.f) return; // warp only matters when stretch is non-zero

    int idxs[JB_MAX_MODES];
    int count = 0;
    for (int i = 0; i < n_modes; ++i){
        if (base[i].active && base[i].keytrack){
            idxs[count++] = i;
        }
    }
    if (count <= 1) return;

    // Use the modal sequence order as the spectral ordering.
    // (If your preset ratios are already ascending, this is stable and avoids resorting artifacts.)

    // Anchor to the lowest eligible mode.
    int id0 = idxs[0];
    float r0 = m[id0].ratio_now;
    if (r0 < 0.000001f) r0 = 0.000001f;
    float d0 = log2f(r0);

    // Warp exponent: p in [0.25..4] when w in [-1..1]
    const float W = 2.f;
    float p = exp2f(w * W);

    for (int rank = 0; rank < count; ++rank){
        int id = idxs[rank];
        float r = m[id].ratio_now;
        if (r < 0.000001f) r = 0.000001f;

        float d = log2f(r);
        float d_rel = d - d0;

        float u = (count > 1) ? ((float)rank / (float)(count - 1)) : 0.f;
        if (u < 0.f) u = 0.f;
        if (u > 1.f) u = 1.f;

        float b = (u <= 0.f) ? 0.f : powf(u, p);
        // multiplier for spacing in log space
        float mult = 1.f + s * b;

        // Safety clamp: prevent negative spacing scale.
        if (mult < 0.f) mult = 0.f;
        if (mult > 4.f) mult = 4.f;

        float d_new = d0 + d_rel * mult;
        float r_new = exp2f(d_new);

        if (r_new < 0.01f) r_new = 0.01f;
        m[id].ratio_now = r_new;
    }
}

static void jb_apply_odd_even_skew_generic(int n_modes, const jb_mode_base_t *base, float odd_skew, float even_skew, jb_mode_rt_t *m){
    const float intensity = 1.0f; // max skew in octaves when skew=±1
    float os = jb_clamp(odd_skew,  -1.f, 1.f);
    float es = jb_clamp(even_skew, -1.f, 1.f);
    if (os == 0.f && es == 0.f) return;

    const float odd_fac  = exp2f(os * intensity);
    const float even_fac = exp2f(es * intensity);

    for (int i = 0; i < n_modes; ++i){
        if (!base[i].active) continue;
        if (!base[i].keytrack) continue;
        const int n = i + 1; // 1-based mode index
        if ((n & 1) == 1) m[i].ratio_now *= odd_fac;
        else              m[i].ratio_now *= even_fac;
    }
}

static void jb_apply_collision_generic(int n_modes, const jb_mode_base_t *base, float collision_amt, jb_mode_rt_t *m){
    float c = jb_clamp(collision_amt, 0.f, 1.f);
    if (c <= 0.f || n_modes <= 2) return;

    float tmp[JB_MAX_MODES];
    for (int i = 0; i < n_modes; ++i){
        tmp[i] = m[i].ratio_now;
    }

    int any_key = 0;
    for (int i = 0; i < n_modes; ++i){
        if (!base[i].active) continue;
        if (!base[i].keytrack) continue;
        any_key = 1;

        int left = i;
        int right = i;

        for (int j = i - 1; j >= 0; --j){
            if (base[j].active && base[j].keytrack){ left = j; break; }
        }
        for (int j = i + 1; j < n_modes; ++j){
            if (base[j].active && base[j].keytrack){ right = j; break; }
        }

        float r_i = m[i].ratio_now;
        float rL = m[left].ratio_now;
        float rR = m[right].ratio_now;
        float avg = 0.5f * (rL + rR);
        tmp[i] = r_i + c * (avg - r_i);
    }

    if (!any_key) return;

    for (int i = 0; i < n_modes; ++i){
        if (!base[i].active) continue;
        if (!base[i].keytrack) continue;
        m[i].ratio_now = tmp[i];
    }
}

// ---------- behavior projection ----------
// ---------- behavior projection ----------
// ---------- behavior projection ----------
static void jb_project_behavior_into_voice_bank(t_juicy_bank_tilde *x, jb_voice_t *v, int bank){
    float xfac = (x->basef0_ref>0.f)? (v->f0 / x->basef0_ref) : 1.f;
    if (xfac < 1e-6f) xfac = 1e-6f;
    v->pitch_x = xfac;

    int n_modes = jb_bank_nmodes(x, bank);

    float *stiffness_add_p    = bank ? &v->stiffness_add2    : &v->stiffness_add;
    float *decay_pitch_mul_p  = bank ? &v->decay_pitch_mul2  : &v->decay_pitch_mul;
    float *decay_vel_mul_p    = bank ? &v->decay_vel_mul2    : &v->decay_vel_mul;
    float *brightness_v_p     = bank ? &v->brightness_v2     : &v->brightness_v;
    float *disp_target_p      = bank ? v->disp_target2       : v->disp_target;

    // Deprecated behavior controls removed: keep neutral multipliers.
    *stiffness_add_p   = 0.f;   // no extra dispersion depth
    *decay_pitch_mul_p = 1.f;   // no pitch-shortening multiplier
    *decay_vel_mul_p   = 1.f;   // no velocity-based decay extension

    // Brightness: user-controlled (+ optional LFO modulation)
    {
        const t_symbol *lfo1_tgt = x->lfo_target[0];
        const t_symbol *lfo2_tgt = x->lfo_target[1];
        const float lfo1 = jb_lfo_value_for_voice(x, v, 0) * jb_clamp(x->lfo_amt_v[0], -1.f, 1.f);
        const float lfo2 = jb_lfo_value_for_voice(x, v, 1) * jb_clamp(x->lfo_amt_eff[1], -1.f, 1.f);
        float b = jb_clamp(jb_bank_brightness(x, bank), -1.f, 1.f);

        float add = 0.f;
        if (lfo1 != 0.f){
            if ((bank == 0 && lfo1_tgt == gensym("brightness_1")) || (bank != 0 && lfo1_tgt == gensym("brightness_2"))){
                add += lfo1;
            }
        }
        if (lfo2 != 0.f){
            if ((bank == 0 && lfo2_tgt == gensym("brightness_1")) || (bank != 0 && lfo2_tgt == gensym("brightness_2"))){
                add += lfo2;
            }
        }
        if (add != 0.f){
            b = jb_clamp(b + add, -1.f, 1.f);
        }
        *brightness_v_p = b;
    }

    // Dispersion targets: keep current quantize-only approach (targets remain 0 => no ratio offsets)
    (void)n_modes;
    for(int i=0;i<n_modes;i++){
        disp_target_p[i] = 0.f;
    }
}


static void jb_project_behavior_into_voice(t_juicy_bank_tilde *x, jb_voice_t *v){
    jb_project_behavior_into_voice_bank(x, v, 0);
}
static void jb_project_behavior_into_voice2(t_juicy_bank_tilde *x, jb_voice_t *v){
    jb_project_behavior_into_voice_bank(x, v, 1);
}

// ---------- update voice coeffs ----------

// ---------- stretch (message only) + apply ----------
static void juicy_bank_tilde_stretch(t_juicy_bank_tilde *x, t_floatarg f){
    float v = jb_clamp(f, -1.f, 1.f);
    if (x->edit_bank) x->stretch2 = v;
    else              x->stretch  = v;
}

static void juicy_bank_tilde_warp(t_juicy_bank_tilde *x, t_floatarg f){
    float v = jb_clamp(f, -1.f, 1.f);
    if (x->edit_bank) x->warp2 = v;
    else              x->warp  = v;
}
static void jb_apply_stretch(const t_juicy_bank_tilde *x, jb_voice_t *v){
    // Legacy wrapper (bank1 only). The actual synthesis path uses jb_apply_stretch_generic()
    // inside jb_project_voice_bank().
    jb_apply_stretch_generic(x->n_modes, x->base, x->stretch, x->warp, v->m);
}

static void jb_apply_collision(const t_juicy_bank_tilde *x, jb_voice_t *v){
    float c = jb_clamp(x->collision_amt, 0.f, 1.f);
    if (c <= 0.f || x->n_modes <= 2)
        return;

    float tmp[JB_MAX_MODES];
    for (int i = 0; i < x->n_modes; ++i){
        tmp[i] = v->m[i].ratio_now;
    }

    int any_key = 0;
    for (int i = 0; i < x->n_modes; ++i){
        if (!x->base[i].active) continue;
        if (!x->base[i].keytrack) continue;
        any_key = 1;

        int left = i;
        int right = i;

        for (int j = i - 1; j >= 0; --j){
            if (x->base[j].active && x->base[j].keytrack){
                left = j;
                break;
            }
        }
        for (int j = i + 1; j < x->n_modes; ++j){
            if (x->base[j].active && x->base[j].keytrack){
                right = j;
                break;
            }
        }

        float r_i = v->m[i].ratio_now;
        float rL = v->m[left].ratio_now;
        float rR = v->m[right].ratio_now;
        float avg = 0.5f * (rL + rR);
        tmp[i] = r_i + c * (avg - r_i);
    }

    if (!any_key)
        return;

    for (int i = 0; i < x->n_modes; ++i){
        if (!x->base[i].active) continue;
        if (!x->base[i].keytrack) continue;
        v->m[i].ratio_now = tmp[i];
    }
}

static void jb_update_voice_coeffs_bank(t_juicy_bank_tilde *x, jb_voice_t *v, int bank){
    int n_modes = jb_bank_nmodes(x, bank);
    const jb_mode_base_t *base = jb_bank_base(x, bank);
    jb_mode_rt_t *m = bank ? v->m2 : v->m;
    float *disp_offset = bank ? v->disp_offset2 : v->disp_offset;
    float *disp_target = bank ? v->disp_target2 : v->disp_target;
    // copy targets to offsets (instant morph for now)
    for(int i=0;i<n_modes;i++){ disp_offset[i] = disp_target[i]; }

    // ratio transforms
    {
        float density_amt_eff = jb_bank_density_amt(x, bank);
        jb_apply_density_generic(n_modes, base, density_amt_eff, m);
    }
jb_lock_fundamental_generic(n_modes, base, m);
    jb_apply_odd_even_skew_generic(n_modes, base, jb_bank_odd_skew(x, bank), jb_bank_even_skew(x, bank), m);
    jb_apply_stretch_generic(n_modes, base, jb_bank_stretch_amt(x, bank), jb_bank_warp_amt(x, bank), m);
    jb_apply_collision_generic(n_modes, base, jb_bank_collision_amt(x, bank), m);

    // --- pitch base (bank semitone + pitch-mod from matrix) ---
    float f0_eff = v->f0;
    if (f0_eff <= 0.f) f0_eff = x->basef0_ref;
    if (f0_eff <= 0.f) f0_eff = 1.f;

    // apply per-bank octave transpose
    int oct = x->bank_octave[bank];
    if (oct != 0){
        f0_eff = ldexpf(f0_eff, oct);
    }

    // apply per-bank semitone transpose
    float semi = (float)x->bank_semitone[bank];
    if (semi != 0.f){
        f0_eff *= powf(2.f, semi / 12.f);
    }

    // apply per-bank cents detune (100 cents = 1 semitone)
    float cents = x->bank_tune_cents[bank];
    if (cents != 0.f){
        f0_eff *= powf(2.f, cents / 1200.f);
    }

// NEW MOD LANES (LFO1/LFO2): bank pitch modulation in Hz, ±1 octave max depth
{
    const t_symbol *lfo1_tgt = x->lfo_target[0];
    const t_symbol *lfo2_tgt = x->lfo_target[1];
    const float lfo1 = jb_lfo_value_for_voice(x, v, 0) * jb_clamp(x->lfo_amt_v[0], -1.f, 1.f);
    const float lfo2 = jb_lfo_value_for_voice(x, v, 1) * jb_clamp(x->lfo_amt_eff[1], -1.f, 1.f);
    float add = 0.f;
    if (lfo1 != 0.f){
        if ((bank == 0 && lfo1_tgt == gensym("pitch_1")) || (bank != 0 && lfo1_tgt == gensym("pitch_2"))){
            add += lfo1;
        }
    }
    if (lfo2 != 0.f){
        if ((bank == 0 && lfo2_tgt == gensym("pitch_1")) || (bank != 0 && lfo2_tgt == gensym("pitch_2"))){
            add += lfo2;
        }
    }
    if (add != 0.f){
        // add is in [-2..+2] worst case, but each lane amount is clamped and targets are unique by lane,
        // so typical is [-1..+1]. Clamp anyway for safety.
        add = jb_clamp(add, -1.f, 1.f);
        f0_eff *= powf(2.f, add);
    }
}

    float (*mm)[JB_N_MODTGT] = jb_bank_mod_matrix(x, bank);

    float pitch_mod = 0.f;
    pitch_mod += x->lfo_val[0] * mm[3][12];
    pitch_mod += x->lfo_val[1] * mm[4][12];
    if (pitch_mod != 0.f){
        float semis = pitch_mod * JB_PITCH_MOD_SEMITONES;
        float ratio = powf(2.f, semis / 12.f);
        f0_eff *= ratio;
    }

        // --- GPD damping (solve once per change) ---

float md_amt = jb_clamp(jb_bank_micro_detune(x, bank),0.f,1.f);

    float disp = jb_bank_dispersion(x, bank);

    for(int i=0;i<n_modes;i++){
        jb_mode_rt_t *md=&m[i];
        if(!base[i].active){
            jb_svf_reset(&md->svfL);
            jb_svf_reset(&md->svfR);
            md->t60_s = 0.f;
            md->nyq_kill = 0;
            continue;
        }

        float ratio_base = md->ratio_now + disp_offset[i];
        // Quantize: snap ratios toward whole integers (0..1).
        if (disp > 0.f){
            float a = jb_clamp(disp, 0.f, 1.f);
            float nearest = roundf(ratio_base);
            ratio_base = (1.f - a)*ratio_base + a*nearest;
        }

        float ratioL = ratio_base;
        float ratioR = ratio_base;
        if(i!=0){ ratioL += md_amt * md->md_hit_offsetL; ratioR += md_amt * md->md_hit_offsetR; }
        if (ratioL < 0.01f) ratioL = 0.01f;
        if (ratioR < 0.01f) ratioR = 0.01f;

        float HzL = base[i].keytrack ? (f0_eff * ratioL) : ratioL;
        float HzR = base[i].keytrack ? (f0_eff * ratioR) : ratioR;
        md->nyq_kill = 0;
        if (HzL >= 0.5f * x->sr || HzR >= 0.5f * x->sr){
            md->nyq_kill = 1;
            HzL = HzR = 0.f;
        } else {
            if (HzL < 0.f) HzL = 0.f;
            if (HzR < 0.f) HzR = 0.f;
        }
        float wL = 2.f * (float)M_PI * HzL / x->sr;
        float wR = 2.f * (float)M_PI * HzR / x->sr;

        if (md->nyq_kill){
            // Nyquist-killed partials are hard-muted and their filter state is cleared
            jb_svf_reset(&md->svfL);
            jb_svf_reset(&md->svfR);
            md->t60_s = 0.f;
            continue;
        }

        // --- Type-4 Caughey damping baseline (frequency-dependent) ---
        // omega is rad/sec (not rad/sample)
        const float LN1000 = 6.907755278982137f; // ln(1000)
        float Hz = 0.5f * (HzL + HzR);
        float omega = 2.f * (float)M_PI * Hz;
        float zeta = jb_bell_zeta_eval(x, v, bank, omega);
        // Convert damping ratio to T60: exp(-zeta*omega*T60) = 1/1000
        float T60 = (zeta <= 1e-9f || omega <= 1e-9f) ? 1e9f : (LN1000 / (zeta * omega));

        float decay_pitch_mul = bank ? v->decay_pitch_mul2 : v->decay_pitch_mul;
        float decay_vel_mul   = bank ? v->decay_vel_mul2   : v->decay_vel_mul;

        T60 *= decay_pitch_mul;
        T60 *= decay_vel_mul;
        md->t60_s = T60;

        // Map T60 -> SVF damping parameter R (Zavalishin):
        // analog envelope: exp(-R*omega*t) ; so R = ln(1000) / (omega*T60)
        float R = (T60 <= 1e-9f || omega <= 1e-9f) ? 0.f : (LN1000 / (omega * T60));
        if (R < 0.f) R = 0.f;
        if (R > 0.2f) R = 0.2f;

        // ZDF SVF tuning variables per channel: g = tan(w/2), w = 2*pi*f/Fs
        float gL = tanf(0.5f * wL);
        float gR = tanf(0.5f * wR);
        if (!isfinite(gL) || gL < 0.f) gL = 0.f;
        if (!isfinite(gR) || gR < 0.f) gR = 0.f;

        jb_svf_set_params(&md->svfL, gL, R);
        jb_svf_set_params(&md->svfR, gR, R);
    }
}
static void jb_update_voice_coeffs(t_juicy_bank_tilde *x, jb_voice_t *v){
    jb_update_voice_coeffs_bank(x, v, 0);
}
static void jb_update_voice_coeffs2(t_juicy_bank_tilde *x, jb_voice_t *v){
    jb_update_voice_coeffs_bank(x, v, 1);
}

// ---------- update voice gains ----------
static void jb_update_voice_gains_bank(const t_juicy_bank_tilde *x, jb_voice_t *v, int bank){
    int n_modes = jb_bank_nmodes(x, bank);
    int active_modes = jb_bank_active_modes(x, bank);
    const jb_mode_base_t *base = jb_bank_base(x, bank);
    jb_mode_rt_t *m = bank ? v->m2 : v->m;
    float *disp_offset = bank ? v->disp_offset2 : v->disp_offset;
    // NEW MOD LANES: LFO direct-to-target modulation values
    const t_symbol *lfo1_tgt = x->lfo_target[0];
    const t_symbol *lfo2_tgt = x->lfo_target[1];
    const float lfo1 = x->lfo_val[0] * jb_clamp(x->lfo_amt_v[0], -1.f, 1.f);
    const float lfo2 = x->lfo_val[1] * jb_clamp(x->lfo_amt_eff[1], -1.f, 1.f);
    // Odd vs Even emphasis bias (-1..+1): index-based mode gain mask
    const float odd_even = jb_clamp(bank ? x->odd_even_bias2 : x->odd_even_bias, -1.f, 1.f);


    // LFO1 -> partials_* : smooth float gating across active modes (0..active_count_idx)
    int   lfo1_partials_k = -1;
    float lfo1_partials_frac = 0.f;
    int   lfo1_partials_enabled = 0;
    int   rank_of_id[JB_MAX_MODES];

    // index-rank of each active mode (0..active_count_idx-1), in ascending mode-ID order
    int   active_count_idx = 0;
    for (int i = 0; i < n_modes; ++i){
        rank_of_id[i] = -1;
        if (!base[i].active) continue;
        rank_of_id[i] = active_count_idx++;
    }

// Elements-style position/pickup weighting (1D), with fixed stereo offsets:
    // Base formula:
    //   w_i = sin(pi * fi * pos) * sin(pi * fi * pickup)
    // We apply a small fixed L/R offset to decorrelate channels ("more HD / wider"),
    // then RMS-normalize per channel to keep level stable.
    float pos_base = (v->velmap_pos[bank] >= 0.f) ? v->velmap_pos[bank] : jb_bank_excite_pos(x, bank);
    float pickup_base = (v->velmap_pickup[bank] >= 0.f) ? v->velmap_pickup[bank] : jb_bank_pickup_pos(x, bank);
    float pos    = jb_clamp(pos_base, 0.f, 1.f);
    float pickup = jb_clamp(pickup_base, 0.f, 1.f);

    // LFO1/LFO2 -> position / pickup (0..1): additive + clamp.
    {
        float add_pos = 0.f;
        float add_pick = 0.f;
        if (lfo1 != 0.f){
            if (lfo1_tgt == gensym("position") || (bank==0 && lfo1_tgt == gensym("position_1")) || (bank!=0 && lfo1_tgt == gensym("position_2"))) add_pos += lfo1;
            else if (lfo1_tgt == gensym("pickup") || (bank==0 && lfo1_tgt == gensym("pickup_1")) || (bank!=0 && lfo1_tgt == gensym("pickup_2"))) add_pick += lfo1;
        }
        if (lfo2 != 0.f){
            if (lfo2_tgt == gensym("position") || (bank==0 && lfo2_tgt == gensym("position_1")) || (bank!=0 && lfo2_tgt == gensym("position_2"))) add_pos += lfo2;
            else if (lfo2_tgt == gensym("pickup") || (bank==0 && lfo2_tgt == gensym("pickup_1")) || (bank!=0 && lfo2_tgt == gensym("pickup_2"))) add_pick += lfo2;
        }
        if (add_pos != 0.f) pos = jb_clamp(pos + add_pos, 0.f, 1.f);
        if (add_pick != 0.f) pickup = jb_clamp(pickup + add_pick, 0.f, 1.f);
    }


    // Small fixed offsets (0..1 domain). Keep subtle to avoid obvious detuning.
    const float pos_off    = 0.004f;
    const float pickup_off = 0.006f;

    const float posL    = jb_clamp(pos    - pos_off,    0.f, 1.f);
    const float posR    = jb_clamp(pos    + pos_off,    0.f, 1.f);
    const float pickupL = jb_clamp(pickup - pickup_off, 0.f, 1.f);
    const float pickupR = jb_clamp(pickup + pickup_off, 0.f, 1.f);

    const float PI = (float)M_PI;
    float energy_posL = 0.f;
    float energy_posR = 0.f;
    for (int i = 0; i < n_modes; ++i){
        if (!base[i].active) continue;
        if (m[i].nyq_kill) continue;
        float ratio = m[i].ratio_now + disp_offset[i];
        float fi = base[i].keytrack ? ratio : ((v->f0 > 0.f) ? (ratio / v->f0) : ratio);
        if (fi < 0.f) fi = 0.f;
        const float wL = sinf(PI * fi * posL) * sinf(PI * fi * pickupL);
        const float wR = sinf(PI * fi * posR) * sinf(PI * fi * pickupR);
        energy_posL += wL * wL;
        energy_posR += wR * wR;
    }
    const float pos_normL = 1.f / sqrtf(energy_posL + 1e-5f);
    const float pos_normR = 1.f / sqrtf(energy_posR + 1e-5f);


float brightness_v = bank ? v->brightness_v2 : v->brightness_v;

    // Tela-style brightness normalization: keep loudness stable when brightness changes.
    // We normalize the summed per-mode gains to match the reference spectrum at brightness=0 (saw slope).
    float sum_gain = 0.f;
    float sum_ref  = 0.f;

    // LFO -> partials_* : smooth float gating across active modes (per-bank)
    {
        float mod = 0.f;
        if (lfo1 != 0.f){
            if ((bank == 0 && lfo1_tgt == gensym("partials_1")) || (bank != 0 && lfo1_tgt == gensym("partials_2"))){
                mod += lfo1;
            }
        }
        if (lfo2 != 0.f){
            if ((bank == 0 && lfo2_tgt == gensym("partials_1")) || (bank != 0 && lfo2_tgt == gensym("partials_2"))){
                mod += lfo2;
            }
        }
        if (mod != 0.f){
            mod = jb_clamp(mod, -1.f, 1.f);
            float pf = (float)active_modes + mod * (float)((active_count_idx > 1) ? (active_count_idx - 1) : 1);
            if (pf < 0.f) pf = 0.f;
            if (pf > (float)active_count_idx) pf = (float)active_count_idx;
            lfo1_partials_k = (int)floorf(pf);
            lfo1_partials_frac = pf - (float)lfo1_partials_k;
            lfo1_partials_enabled = 1;
        }
    }

    for(int i = 0; i < n_modes; ++i){
        jb_mode_rt_t *md = &m[i];
        if(!base[i].active){
            jb_svf_reset(&md->svfL);
            jb_svf_reset(&md->svfR);
            md->t60_s = 0.f;
            md->nyq_kill = 0;
            continue;
        }

        float ratio = m[i].ratio_now + disp_offset[i];
        float ratio_rel = base[i].keytrack ? ratio : ((v->f0>0.f)? (ratio / v->f0) : ratio);

        float g = base[i].base_gain * jb_bright_gain(ratio_rel, brightness_v);

        float g_ref = base[i].base_gain * jb_bright_gain(ratio_rel, 0.f);
        float w = 1.f;
        float gn = g * w;
        float gn_ref = g_ref * w;
        if (m[i].nyq_kill) { gn = 0.f; gn_ref = 0.f; gn_ref = 0.f; }

        if (active_modes <= 0){
            gn = 0.f; gn_ref = 0.f;
        } else if (active_modes < n_modes){
            int K = active_modes;
            if (i >= K){
                float fade_width = 3.f;
                float u = ((float)i - (float)K) / fade_width;
                if (u >= 1.f){
                    gn = 0.f; gn_ref = 0.f;
                } else if (u > 0.f){
                    float w_fade = 0.5f * (1.f + cosf((float)M_PI * u));
                    gn *= w_fade;
                    gn_ref *= w_fade;
                }
            }
        }

        // LFO1 partials gating (smooth float: 0..32)

        if (lfo1_partials_enabled){

            const int r = rank_of_id[i];

            float w_p = 0.f;

            if (r < lfo1_partials_k) w_p = 1.f;

            else if (r == lfo1_partials_k) w_p = lfo1_partials_frac;

            else w_p = 0.f;

            gn *= w_p;

            gn_ref *= w_p;

        }
        // --- Position/Pickup mask (1D, Elements-style; signed) ---
        float gnL = gn, gnR = gn;
        float gn_refL = gn_ref, gn_refR = gn_ref;
        {
            float ratio_p = m[i].ratio_now + disp_offset[i];
            float fi = base[i].keytrack ? ratio_p : ((v->f0 > 0.f) ? (ratio_p / v->f0) : ratio_p);
            if (fi < 0.f) fi = 0.f;
            const float wL = (sinf(PI * fi * posL) * sinf(PI * fi * pickupL)) * pos_normL;
            const float wR = (sinf(PI * fi * posR) * sinf(PI * fi * pickupR)) * pos_normR;
            gnL     *= wL;
            gnR     *= wR;
            gn_refL *= wL;
            gn_refR *= wR;
        }

        // Odd vs Even mask by mode index (1-based): odd => (1-bias), even => (1+bias)
        const int mode_num = i + 1;
        float oe = (mode_num & 1) ? (1.f - odd_even) : (1.f + odd_even);
        oe = jb_clamp(oe, 0.f, 1.f);
        gnL     *= oe;
        gnR     *= oe;

        m[i].gain_nowL = gnL;
        m[i].gain_nowR = gnR;
        sum_gain += 0.5f * (fabsf(gnL) + fabsf(gnR));
        sum_ref  += 0.5f * (fabsf(gn_refL) + fabsf(gn_refR));
    }

    // Apply normalization so brightness redistributes energy without changing overall level.
    float norm = (sum_gain > 1e-12f) ? (sum_ref / sum_gain) : 1.f;
    if (norm < 0.f) norm = 0.f;
    for (int i = 0; i < n_modes; ++i){
        m[i].gain_nowL *= norm;
        m[i].gain_nowR *= norm;
    }
}

static void jb_update_voice_gains(const t_juicy_bank_tilde *x, jb_voice_t *v){
    jb_update_voice_gains_bank(x, v, 0);
}
static void jb_update_voice_gains2(const t_juicy_bank_tilde *x, jb_voice_t *v){
    jb_update_voice_gains_bank(x, v, 1);
}
static void jb_voice_reset_states(const t_juicy_bank_tilde *x, jb_voice_t *v, jb_rng_t *rng){
    (void)rng;

    // Release envelopes (used for smooth note-off / voice stealing)
    v->rel_env  = 1.f;
    v->rel_env2 = 1.f;

    // Optional energy meter (not used for stealing anymore, keep sane)
    v->energy = 0.f;

    // Per-voice one-shot LFO state
    for (int li = 0; li < JB_N_LFO; ++li){
        v->lfo_phase_state[li] = 0.f;
        v->lfo_val[li]         = 0.f;
        v->lfo_snh[li]         = 0.f;
        v->lfo_oneshot_done[li]= 0;
    }
    // Exciter runtime
    jb_exc_voice_reset_runtime(&v->exc);

    // Velocity-mapping overrides
    for (int b = 0; b < 2; ++b){
        for (int d = 0; d < JB_N_DAMPERS; ++d){
            v->velmap_bell_zeta[b][d] = -1.f;
        }
    }

    // Reset per-mode runtime state for both banks
    for (int i = 0; i < x->n_modes; ++i){
        v->m[i].svfL.s1 = v->m[i].svfL.s2 = 0.f;
        v->m[i].svfR.s1 = v->m[i].svfR.s2 = 0.f;
        v->m[i].y_pre_lastL = v->m[i].y_pre_lastR = 0.f;
    }
    for (int i = 0; i < x->n_modes2; ++i){
        v->m2[i].svfL.s1 = v->m2[i].svfL.s2 = 0.f;
        v->m2[i].svfR.s1 = v->m2[i].svfR.s2 = 0.f;
        v->m2[i].y_pre_lastL = v->m2[i].y_pre_lastR = 0.f;
    }
}


static int jb_find_voice_to_steal(t_juicy_bank_tilde *x){
    // Prefer an IDLE voice; otherwise steal the quietest RELEASE voice.
    int best_idle = -1;
    int best_rel  = -1;
    float best_rel_env = 1e9f;

    for(int i=0;i<x->max_voices;i++){
        const jb_voice_t *v = &x->v[i];
        if(v->state == V_IDLE){
            best_idle = i;
            break;
        }
        if(v->state == V_RELEASE){
            const float e = (v->rel_env < v->rel_env2 ? v->rel_env : v->rel_env2);
            if(e < best_rel_env){
                best_rel_env = e;
                best_rel = i;
            }
        }
    }
    if(best_idle >= 0) return best_idle;
    if(best_rel  >= 0) return best_rel;

    // Worst case: steal voice 0 (all voices held)
    return 0;
}


// ---------- INTERNAL EXCITER (Fusion STEP 2) — note triggers ----------

// Hard reset used only as a "panic" guard if a voice goes unstable (NaN/INF/runaway).
// This does NOT change normal sound; it only triggers on broken states.
static void jb_voice_panic_reset(t_juicy_bank_tilde *x, jb_voice_t *v){
    if (!x || !v) return;

    v->state = V_IDLE;
    v->rel_env = 0.f;
    v->rel_env2 = 0.f;
    v->energy = 0.f;
    // Feedback loop state (per bank)
    for (int b = 0; b < 2; ++b){
    }

    // Internal exciter reset
    jb_exc_voice_reset_runtime(&v->exc);


    // Clear resonator runtime states (keep ratios/decays/gains as-is so next note is consistent)
    for (int i = 0; i < x->n_modes; ++i){
        jb_mode_rt_t *md = &v->m[i];
        jb_svf_reset(&md->svfL);
        jb_svf_reset(&md->svfR);
        md->driveL = md->driveR = 0.f;
        md->y_pre_lastL = md->y_pre_lastR = 0.f;
        md->hit_gateL = md->hit_gateR = 0;
        md->hit_coolL = md->hit_coolR = 0;
        md->nyq_kill = 0;
    }
    for (int i = 0; i < x->n_modes2; ++i){
        jb_mode_rt_t *md = &v->m2[i];
        jb_svf_reset(&md->svfL);
        jb_svf_reset(&md->svfR);
        md->driveL = md->driveR = 0.f;
        md->y_pre_lastL = md->y_pre_lastR = 0.f;
        md->hit_gateL = md->hit_gateR = 0;
        md->hit_coolL = md->hit_coolR = 0;
        md->nyq_kill = 0;
    }
}

static inline void jb_exc_note_on(t_juicy_bank_tilde *x, jb_voice_t *v, float vel){
    jb_exc_voice_t *e = &v->exc;
    e->vel_cur = jb_clamp(vel, 0.f, 1.f);
    if (e->vel_cur > 0.f){
        e->vel_on = e->vel_cur;
        jb_exc_adsr_note_on(&e->env);

        // tiny stereo variation per strike
        e->gainL = 1.f + 0.02f * jb_exc_noise_tpdf(&e->rngL);
        e->gainR = 1.f + 0.02f * jb_exc_noise_tpdf(&e->rngR);

        jb_exc_pulse_trigger(&e->pulseL);
        jb_exc_pulse_trigger(&e->pulseR);
    }else{
        jb_exc_adsr_note_off(&e->env);
    }
}

static inline void jb_exc_note_off(jb_voice_t *v){
    jb_exc_adsr_note_off(&v->exc.env);
}
// ---- forward declarations (avoid C99 implicit declarations) ----
static void jb_apply_velocity_mapping(t_juicy_bank_tilde *x, jb_voice_t *v);
static inline int jb_target_is_none(t_symbol *s);
static inline float jb_bell_map_norm_to_zeta(float u);

static inline void jb_start_note_on_voice(t_juicy_bank_tilde *x, jb_voice_t *v, float f0, float vel01){
    v->pending_note = 0;
    v->steal_countdown = 0;

    v->state = V_HELD;
    v->f0  = (f0<=0.f)?1.f:f0;
    v->vel = jb_clamp(vel01, 0.f, 1.f);

    jb_voice_reset_states(x, v, &x->rng);
    jb_apply_velocity_mapping(x, v);

    // One-shot LFO init (per-note)
    for (int li = 0; li < JB_N_LFO; ++li){
        int mode = (int)floorf(x->lfo_mode_v[li] + 0.5f);
        if (mode != 2) continue;

        int shape = (int)floorf(x->lfo_shape_v[li] + 0.5f);
        if (shape < 1) shape = 1;
        if (shape > 5) shape = 5;

        v->lfo_phase_state[li] = 0.f;
        v->lfo_oneshot_done[li] = 0;

        if (shape == 4){
            v->lfo_val[li] = jb_rng_bi(&x->rng);
            v->lfo_snh[li] = v->lfo_val[li];
            v->lfo_oneshot_done[li] = 1;
        } else {
            float ph = jb_clamp(x->lfo_phase_v[li], 0.f, 1.f);
            float val = 0.f;
            if (shape == 1)      val = 2.f * ph - 1.f;
            else if (shape == 5) val = 1.f - 2.f * ph;
            else if (shape == 2) val = (ph < 0.5f) ? 1.f : -1.f;
            else                 val = sinf(2.f * M_PI * ph);
            v->lfo_val[li] = val;
        }
    }

    jb_project_behavior_into_voice(x, v);
    jb_project_behavior_into_voice2(x, v);
    jb_exc_note_on(x, v, v->vel);
}


static void jb_note_on(t_juicy_bank_tilde *x, float f0, float vel){
    // Pick a voice among the playable pool (0..max_voices-1).
    int idx = jb_find_voice_to_steal(x);
    jb_voice_t *v = &x->v[idx];

    if (f0 <= 0.f) f0 = 1.f;
    // Velocity is accepted as either 0..1 or MIDI 0..127, and always stored 0..1.
    vel = jb_exc_midi_to_vel01(vel);

    // If this voice is currently active, do a very short fade-out before restarting it
    // to avoid hard discontinuities.
    if (v->state != V_IDLE){
        v->pending_note = 1;
        v->pending_f0   = f0;
        v->pending_vel  = vel;
        v->steal_countdown = JB_STEAL_FADE_SAMPLES;

        jb_exc_note_off(v);
        v->state = V_RELEASE;
        // Ensure the release envelope starts "on" so we actually fade.
        if (v->rel_env  < 1e-6f) v->rel_env  = 1.f;
        if (v->rel_env2 < 1e-6f) v->rel_env2 = 1.f;
        return;
    }

    // Start (or restart) immediately
    v->pending_note = 0;
    v->steal_countdown = 0;

    v->state = V_HELD;
    v->f0  = f0;
    v->vel = vel;

    jb_voice_reset_states(x, v, &x->rng);
    jb_apply_velocity_mapping(x, v);

    // One-shot LFO init (per-note): reset phase and (for S&H) pick a fresh random value once.
    for (int li = 0; li < JB_N_LFO; ++li){
        int mode = (int)floorf(x->lfo_mode_v[li] + 0.5f);
        if (mode != 2) continue;

        int shape = (int)floorf(x->lfo_shape_v[li] + 0.5f);
        if (shape < 1) shape = 1;
        if (shape > 5) shape = 5;

        v->lfo_phase_state[li] = 0.f;
        v->lfo_oneshot_done[li] = 0;

        if (shape == 4){
            // S&H one-shot: one random value per note-on, held forever.
            v->lfo_val[li] = jb_rng_bi(&x->rng);
            v->lfo_snh[li] = v->lfo_val[li];
            v->lfo_oneshot_done[li] = 1;
        } else {
            float ph = jb_clamp(x->lfo_phase_v[li], 0.f, 1.f);
            float val = 0.f;
            if (shape == 1)      val = 2.f * ph - 1.f;
            else if (shape == 5) val = 1.f - 2.f * ph;
            else if (shape == 2) val = (ph < 0.5f) ? 1.f : -1.f;
            else                 val = sinf(2.f * M_PI * ph);
            v->lfo_val[li] = val;
        }
    }

    jb_project_behavior_into_voice(x, v);
    jb_project_behavior_into_voice2(x, v);
    jb_exc_note_on(x, v, v->vel);
}


// ---------- Velocity mapping lane (per-note) ----------
static void jb_apply_velocity_mapping(t_juicy_bank_tilde *x, jb_voice_t *v){
    if (!x || !v) return;

    float amt = jb_clamp(x->velmap_amount, -1.f, 1.f);
    if (amt == 0.f) return;

    float vel = jb_clamp(v->vel, 0.f, 1.f);
    float delta = amt * vel; // bipolar scaling shared by all enabled targets

    // Clear any previous per-note overrides (voice might be reused)
    for (int b = 0; b < 2; ++b){
        v->velmap_pos[b] = -1.f;
        v->velmap_pickup[b] = -1.f;
        v->velmap_master_add[b] = 0.f;
        v->velmap_brightness_add[b] = 0.f;
        for (int d = 0; d < JB_N_DAMPERS; ++d){
            v->velmap_bell_zeta_on[b][d] = 0;
            v->velmap_bell_zeta[b][d] = 0.f;
        }
    }
    // Exciter per-note overrides
    v->exc.imp_shape_v = -1.f;
    v->exc.noise_timbre_v = -1.f;
    v->exc.a_ms_v = -1.f;
    v->exc.d_ms_v = -1.f;
    v->exc.r_ms_v = -1.f;

    // Apply every enabled velocity-mapping target (toggles).
    for (int ti = 0; ti < JB_VELMAP_N_TARGETS; ++ti){
        if (!x->velmap_on[ti]) continue;

        switch((jb_velmap_idx)ti){

            // ---- Bell damper zeta peaks ----
            case JB_VEL_BELL_Z_D1_B1:
            case JB_VEL_BELL_Z_D1_B2:
            case JB_VEL_BELL_Z_D2_B1:
            case JB_VEL_BELL_Z_D2_B2:
            case JB_VEL_BELL_Z_D3_B1:
            case JB_VEL_BELL_Z_D3_B2: {
                int damper = ((ti - JB_VEL_BELL_Z_D1_B1) / 2); // 0..2
                int bank   = ((ti - JB_VEL_BELL_Z_D1_B1) % 2); // 0..1
                if (damper >= 0 && damper < JB_N_DAMPERS){
                    float u = x->bell_peak_zeta_param[bank][damper];
                    if (u < 0.f) u = 0.5915f; // sensible on-default
                    u = jb_clamp(u + delta, 0.f, 1.f);
                    v->velmap_bell_zeta_on[bank][damper] = 1;
                    v->velmap_bell_zeta[bank][damper] = jb_bell_map_norm_to_zeta(u);
                }
            } break;

            case JB_VEL_BRIGHTNESS_1: v->velmap_brightness_add[0] = delta; break;
            case JB_VEL_BRIGHTNESS_2: v->velmap_brightness_add[1] = delta; break;

            case JB_VEL_POSITION_1: {
                float base = jb_clamp(jb_bank_excite_pos(x, 0), 0.f, 1.f);
                v->velmap_pos[0] = jb_clamp(base + delta, 0.f, 1.f);
            } break;
            case JB_VEL_POSITION_2: {
                float base = jb_clamp(jb_bank_excite_pos(x, 1), 0.f, 1.f);
                v->velmap_pos[1] = jb_clamp(base + delta, 0.f, 1.f);
            } break;

            case JB_VEL_PICKUP_1: {
                float base = jb_clamp(jb_bank_pickup_pos(x, 0), 0.f, 1.f);
                v->velmap_pickup[0] = jb_clamp(base + delta, 0.f, 1.f);
            } break;
            case JB_VEL_PICKUP_2: {
                float base = jb_clamp(jb_bank_pickup_pos(x, 1), 0.f, 1.f);
                v->velmap_pickup[1] = jb_clamp(base + delta, 0.f, 1.f);
            } break;

            case JB_VEL_MASTER_1: v->velmap_master_add[0] = delta; break;
            case JB_VEL_MASTER_2: v->velmap_master_add[1] = delta; break;

            // ---- Exciter ADSR (noise source) ----
            case JB_VEL_ADSR_ATTACK: {
                float base = (x->exc_attack_ms > 0.f) ? x->exc_attack_ms : 1.f;
                float ms = base * (1.f + delta);
                if (ms < 0.f) ms = 0.f;
                if (ms > 10000.f) ms = 10000.f;
                v->exc.a_ms_v = ms;
            } break;
            case JB_VEL_ADSR_DECAY: {
                float base = (x->exc_decay_ms > 0.f) ? x->exc_decay_ms : 1.f;
                float ms = base * (1.f + delta);
                if (ms < 0.f) ms = 0.f;
                if (ms > 10000.f) ms = 10000.f;
                v->exc.d_ms_v = ms;
            } break;
            case JB_VEL_ADSR_RELEASE: {
                float base = (x->exc_release_ms > 0.f) ? x->exc_release_ms : 1.f;
                float ms = base * (1.f + delta);
                if (ms < 0.f) ms = 0.f;
                if (ms > 10000.f) ms = 10000.f;
                v->exc.r_ms_v = ms;
            } break;

            case JB_VEL_IMP_SHAPE: {
                float base = jb_clamp(x->exc_imp_shape, 0.f, 1.f);
                v->exc.imp_shape_v = jb_clamp(base + delta, 0.f, 1.f);
            } break;

            case JB_VEL_NOISE_TIMBRE: {
                float base = jb_clamp(x->exc_shape, 0.f, 1.f);
                v->exc.noise_timbre_v = jb_clamp(base + delta, 0.f, 1.f);
            } break;

            default: break;
        }
    }
}

static void jb_note_on_voice(t_juicy_bank_tilde *x, int vix1, float f0, float vel){
    // This path is used by [note_poly]/[poly]-style voice addressing.
    // We still want smooth behavior here: stealing a busy voice fades it out briefly before restarting.
    if (vix1 < 1) vix1 = 1;
    if (vix1 > x->max_voices) vix1 = x->max_voices;
    int idx = vix1 - 1;

    if (f0 <= 0.f) f0 = 1.f;
    // Velocity is accepted as either 0..1 or MIDI 0..127.
    vel = jb_exc_midi_to_vel01(vel);

    jb_voice_t *v = &x->v[idx];
    // If this voice is currently active, do a short fade-out before restarting it.
    if (v->state != V_IDLE){
        v->pending_note = 1;
        v->pending_f0   = f0;
        v->pending_vel  = vel;
        v->steal_countdown = JB_STEAL_FADE_SAMPLES;
        jb_exc_note_off(v);
        v->state = V_RELEASE;
        if (v->rel_env  < 1e-6f) v->rel_env  = 1.f;
        if (v->rel_env2 < 1e-6f) v->rel_env2 = 1.f;
        return;
    }

    // Start the new note immediately in the requested attack slot.
    v->state = V_HELD;
    v->f0 = f0;
    v->vel = vel;

    jb_voice_reset_states(x, v, &x->rng);
    jb_apply_velocity_mapping(x, v);
    jb_project_behavior_into_voice(x, v);
    jb_project_behavior_into_voice2(x, v);
    jb_exc_note_on(x, v, v->vel);
}

static void jb_note_off_voice(t_juicy_bank_tilde *x, int vix1){
    // Voice-index only (legacy / hard off). Kept for off_poly.
    if (vix1 < 1) vix1 = 1;
    if (vix1 > x->max_voices) vix1 = x->max_voices;
    int idx = vix1 - 1;
    if (x->v[idx].state != V_IDLE){
        jb_exc_note_off(&x->v[idx]);
        x->v[idx].state = V_RELEASE;
    }
}

static void jb_note_off_voice_pitch(t_juicy_bank_tilde *x, int vix1, float f0){
    // Safer NOTE-OFF for voice-addressed poly:
    // Only releases if the pitch matches the current voice pitch (prevents old note-offs
    // from killing a newer note after voice stealing).
    if (vix1 < 1) vix1 = 1;
    if (vix1 > x->max_voices) vix1 = x->max_voices;
    int idx = vix1 - 1;

    jb_voice_t *v = &x->v[idx];
    if (v->state == V_IDLE) return;

    if (f0 > 0.f && v->f0 > 0.f){
        float ratio = v->f0 / f0;
        if (ratio < 1.f) ratio = 1.f / ratio;

        // ~20 cents tolerance
        const float tol = 1.0116f;
        if (ratio > tol){
            return; // ignore mismatched note-off
        }
    }

    jb_exc_note_off(v);
    v->state = V_RELEASE;
}

// Message handlers (voice-addressed)
static void juicy_bank_tilde_note_poly(t_juicy_bank_tilde *x, t_floatarg vix, t_floatarg f0, t_floatarg vel){
    if (vel <= 0.f) { jb_note_off_voice_pitch(x, (int)vix, f0); }
    else            { jb_note_on_voice(x, (int)vix, f0, vel); }
}

static void juicy_bank_tilde_note_poly_midi(t_juicy_bank_tilde *x, t_floatarg vix, t_floatarg midi, t_floatarg vel){
    float f0 = jb_midi_to_hz(midi);
    if (vel <= 0.f) { jb_note_off_voice_pitch(x, (int)vix, f0); }
    else            { jb_note_on_voice(x, (int)vix, f0, vel); }
}

// ---------- perform ----------

static t_int *juicy_bank_tilde_perform(t_int *w){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)(w[1]);
    // outputs
    t_sample *outL=(t_sample *)(w[2]);
    t_sample *outR=(t_sample *)(w[3]);
    int n=(int)(w[4]);
#if defined(__SSE__) || defined(__SSE2__)
    // Avoid denormal/subnormal slowdowns on Intel CPUs (does not affect audible quality).
    _MM_SET_FLUSH_ZERO_MODE(_MM_FLUSH_ZERO_ON);
  #if defined(__SSE3__)
    _MM_SET_DENORMALS_ZERO_MODE(_MM_DENORMALS_ZERO_ON);
  #endif
#endif

    /* ------------------ EARLY-OUT (CPU) ------------------
   If there are no active voices, we can zero the outputs and return immediately.
   This is a big win on Bela when idle.
------------------------------------------------------- */
    int any_active = 0;
    for(int v=0; v<x->max_voices; ++v){
        const jb_voice_t *V = &x->v[v];
        if(V->state != V_IDLE || V->rel_env > 1e-6f || V->rel_env2 > 1e-6f){
            any_active = 1;
            break;
        }
    }
    if(!any_active){
        for(int i=0;i<n;i++){ outL[i]=0; outR[i]=0; }
        return (w + 5);
    }

// clear outputs
    for(int i=0;i<n;i++){ outL[i]=0; outR[i]=0; }

    // update LFOs once per block (for modulation matrix sources)
    jb_update_lfos_block(x, n);    // Cached symbols (initialized in setup()); no gensym() in the audio thread.
    const t_symbol *sym_master_1     = jb_sym_master_1;
    const t_symbol *sym_master_2     = jb_sym_master_2;
    const t_symbol *sym_lfo2_amount  = jb_sym_lfo2_amount;
    const t_symbol *sym_noise_timbre = jb_sym_noise_timbre;
    const t_symbol *sym_imp_shape    = jb_sym_imp_shape;

    // NEW MOD LANES: LFO1 output (scaled by its amount) + a few global mods that must happen pre-exciter-update
    const t_symbol *lfo1_tgt = x->lfo_target[0];
    const float lfo1_amt = jb_clamp(x->lfo_amt_v[0], -1.f, 1.f);
    const float lfo1_out = x->lfo_val[0] * lfo1_amt;

    // store effective LFO amounts for downstream use
    x->lfo_amt_eff[0] = lfo1_amt;
    x->lfo_amt_eff[1] = jb_clamp(x->lfo_amt_v[1], -1.f, 1.f);
    if (lfo1_out != 0.f && lfo1_tgt == sym_lfo2_amount) {
        x->lfo_amt_eff[1] = jb_clamp(x->lfo_amt_eff[1] + lfo1_out, -1.f, 1.f);
    }

    // LFO1/LFO2 -> Exciter params (0..1, additive, clamped) must be applied before jb_exc_update_block()
    const float noise_timbre_saved = x->exc_shape;      // Noise timbre/color (0..1)
    const float imp_shape_saved    = x->exc_imp_shape;  // Impulse-only shape (0..1)

    const t_symbol *lfo2_tgt = x->lfo_target[1];
    const float lfo2_out = x->lfo_val[1] * jb_clamp(x->lfo_amt_eff[1], -1.f, 1.f);

    float noise_timbre = noise_timbre_saved;
    float imp_shape    = imp_shape_saved;

    if (lfo1_out != 0.f){
        if (lfo1_tgt == sym_noise_timbre){
            noise_timbre += lfo1_out;
        } else if (lfo1_tgt == sym_imp_shape){
            imp_shape += lfo1_out;
        }
    }
    if (lfo2_out != 0.f){
        if (lfo2_tgt == sym_noise_timbre){
            noise_timbre += lfo2_out;
        } else if (lfo2_tgt == sym_imp_shape){
            imp_shape += lfo2_out;
        }
    }

    x->exc_shape     = jb_clamp(noise_timbre, 0.f, 1.f);
    x->exc_imp_shape = jb_clamp(imp_shape,    0.f, 1.f);

    // Internal exciter: update shared params -> per-voice filters + ADSR times/curves
    jb_exc_update_block(x);

    // Restore (so parameters remain stable / inspectable on the Pd side)
    x->exc_shape     = noise_timbre_saved;
    x->exc_imp_shape = imp_shape_saved;

        // (Coupling removed) Both banks are always excited by the internal exciter and always summed to the output.
    // Internal exciter mix weights (computed once per block)
    float exc_f = jb_clamp(x->exc_fader, -1.f, 1.f);
    float exc_t = 0.5f * (exc_f + 1.f);
    float exc_w_imp   = sinf(0.5f * (float)M_PI * exc_t);
    float exc_w_noise = cosf(0.5f * (float)M_PI * exc_t);
    // Process per-voice, sample-major so feedback uses only a 2-sample delay (no block latency)
    for(int vix=0; vix<x->total_voices; ++vix){
        jb_voice_t *v = &x->v[vix];
        if (v->state==V_IDLE) continue;
        jb_update_lfos_oneshot_voice_block(x, v, n);

        // Per-bank master gain (0..1) with modulation-matrix target "master" (index 11).
        float (*mm1)[JB_N_MODTGT] = x->mod_matrix;
        float (*mm2)[JB_N_MODTGT] = x->mod_matrix2;

        float master_base1 = x->bank_master[0];
        float master_base2 = x->bank_master[1];

        float master_mod1 = 0.f;
        float master_mod2 = 0.f;
        for (int src = 0; src < JB_N_MODSRC; ++src){
            float src_v = jb_mod_source_value(x, v, src);
            if (src_v == 0.f) continue;
            master_mod1 += mm1[src][11] * src_v;
            master_mod2 += mm2[src][11] * src_v;
        }
        if (master_mod1 > 1.f) master_mod1 = 1.f; else if (master_mod1 < -1.f) master_mod1 = -1.f;
        if (master_mod2 > 1.f) master_mod2 = 1.f; else if (master_mod2 < -1.f) master_mod2 = -1.f;

        float bank_gain1;
        if (master_mod1 >= 0.f) bank_gain1 = master_base1 + master_mod1 * (1.f - master_base1);
        else                    bank_gain1 = master_base1 + master_mod1 * master_base1;

        float bank_gain2;
        if (master_mod2 >= 0.f) bank_gain2 = master_base2 + master_mod2 * (1.f - master_base2);
        else                    bank_gain2 = master_base2 + master_mod2 * master_base2;


    // Release time (per bank): map 0..1 -> ~5ms..2000ms and convert to per-sample exp coefficient
    const float rel_amt1 = jb_bank_release_amt(x, 0);
    const float rel_amt2 = jb_bank_release_amt(x, 1);
    const float rel_ms1  = 5.f + 1995.f * rel_amt1;
    const float rel_ms2  = 5.f + 1995.f * rel_amt2;
    const float a_rel1_sample = expf(-1.f / (rel_ms1 * 0.001f * x->sr));
    const float a_rel2_sample = expf(-1.f / (rel_ms2 * 0.001f * x->sr));

        // LFO1/LFO2 -> master_* (bank output volume): additive + clamp (0..1)
        {
            const t_symbol *lfo2_tgt_local = x->lfo_target[1];
            const float lfo1_out_v = jb_lfo_value_for_voice(x, v, 0) * lfo1_amt;
            const float lfo2_out_local = jb_lfo_value_for_voice(x, v, 1) * jb_clamp(x->lfo_amt_eff[1], -1.f, 1.f);

            float add1 = 0.f;
            float add2 = 0.f;

            if (lfo1_out_v != 0.f){
                if (lfo1_tgt == sym_master_1) add1 += lfo1_out_v;
                else if (lfo1_tgt == sym_master_2) add2 += lfo1_out_v;
            }
            if (lfo2_out_local != 0.f){
                if (lfo2_tgt_local == sym_master_1) add1 += lfo2_out_local;
                else if (lfo2_tgt_local == sym_master_2) add2 += lfo2_out_local;
            }

            if (add1 != 0.f) bank_gain1 = jb_clamp(bank_gain1 + add1, 0.f, 1.f);
            if (add2 != 0.f) bank_gain2 = jb_clamp(bank_gain2 + add2, 0.f, 1.f);
        }

        // Velocity mapping -> per-voice bank gain (additive, clamped)
        if (v->velmap_master_add[0] != 0.f) bank_gain1 = jb_clamp(bank_gain1 + v->velmap_master_add[0], 0.f, 1.f);
        if (v->velmap_master_add[1] != 0.f) bank_gain2 = jb_clamp(bank_gain2 + v->velmap_master_add[1], 0.f, 1.f);

        // Pan is intentionally not used in this synth anymore.

        const jb_mode_base_t *base1 = x->base;
        const jb_mode_base_t *base2 = x->base2;

        for(int i=0;i<n;i++){
            // Per-bank voice outputs (pre-space)
            float b1OutL = 0.f, b1OutR = 0.f;
            float b2OutL = 0.f, b2OutR = 0.f;
            // Exciter (feedback removed)
            float ex0L = 0.f, ex0R = 0.f;
            jb_exc_process_sample(x, v, exc_w_imp, exc_w_noise, &ex0L, &ex0R);
            float exL = ex0L;
            float exR = ex0R;


// -------- BANK 2 --------
            if (bank_gain2 > 0.f && v->rel_env2 > 0.f){
                #if JB_HAVE_NEON && JB_ENABLE_NEON
                int m=0;
                for(; m+3 < x->n_modes2; m+=4){
                    // Early skip if all 4 inactive
                    jb_mode_rt_t *md0=&v->m2[m+0];
                    jb_mode_rt_t *md1=&v->m2[m+1];
                    jb_mode_rt_t *md2=&v->m2[m+2];
                    jb_mode_rt_t *md3=&v->m2[m+3];
                    uint32_t am0 = (base2[m+0].active && !md0->nyq_kill && (fabsf(md0->gain_nowL)+fabsf(md0->gain_nowR))>0.f) ? 0xFFFFFFFFu : 0u;
                    uint32_t am1 = (base2[m+1].active && !md1->nyq_kill && (fabsf(md1->gain_nowL)+fabsf(md1->gain_nowR))>0.f) ? 0xFFFFFFFFu : 0u;
                    uint32_t am2 = (base2[m+2].active && !md2->nyq_kill && (fabsf(md2->gain_nowL)+fabsf(md2->gain_nowR))>0.f) ? 0xFFFFFFFFu : 0u;
                    uint32_t am3 = (base2[m+3].active && !md3->nyq_kill && (fabsf(md3->gain_nowL)+fabsf(md3->gain_nowR))>0.f) ? 0xFFFFFFFFu : 0u;
                    if(!(am0|am1|am2|am3)) continue;
                    uint32x4_t activeMask = (uint32x4_t){am0, am1, am2, am3};
                
                    // Gather SVF params/states (L)
                    float gL_[4]  = {md0->svfL.g,  md1->svfL.g,  md2->svfL.g,  md3->svfL.g};
                    float dL_[4]  = {md0->svfL.d,  md1->svfL.d,  md2->svfL.d,  md3->svfL.d};
                    float s1L_[4] = {md0->svfL.s1, md1->svfL.s1, md2->svfL.s1, md3->svfL.s1};
                    float s2L_[4] = {md0->svfL.s2, md1->svfL.s2, md2->svfL.s2, md3->svfL.s2};
                    float32x4_t gL4  = vld1q_f32(gL_);
                    float32x4_t dL4  = vld1q_f32(dL_);
                    float32x4_t s1L4 = vld1q_f32(s1L_);
                    float32x4_t s2L4 = vld1q_f32(s2L_);
                
                    // Gather SVF params/states (R)
                    float gR_[4]  = {md0->svfR.g,  md1->svfR.g,  md2->svfR.g,  md3->svfR.g};
                    float dR_[4]  = {md0->svfR.d,  md1->svfR.d,  md2->svfR.d,  md3->svfR.d};
                    float s1R_[4] = {md0->svfR.s1, md1->svfR.s1, md2->svfR.s1, md3->svfR.s1};
                    float s2R_[4] = {md0->svfR.s2, md1->svfR.s2, md2->svfR.s2, md3->svfR.s2};
                    float32x4_t gR4  = vld1q_f32(gR_);
                    float32x4_t dR4  = vld1q_f32(dR_);
                    float32x4_t s1R4 = vld1q_f32(s1R_);
                    float32x4_t s2R4 = vld1q_f32(s2R_);
                
                    // Drive update + input vectors
                    float driveL0=md0->driveL, driveL1=md1->driveL, driveL2=md2->driveL, driveL3=md3->driveL;
                    float driveR0=md0->driveR, driveR1=md1->driveR, driveR2=md2->driveR, driveR3=md3->driveR;
                    const float att_a = 1.f;
                    if(am0){ float excL = exL * md0->gain_nowL; float excR = exR * md0->gain_nowR; driveL0 += att_a*(excL-driveL0); driveR0 += att_a*(excR-driveR0); }
                    if(am1){ float excL = exL * md1->gain_nowL; float excR = exR * md1->gain_nowR; driveL1 += att_a*(excL-driveL1); driveR1 += att_a*(excR-driveR1); }
                    if(am2){ float excL = exL * md2->gain_nowL; float excR = exR * md2->gain_nowR; driveL2 += att_a*(excL-driveL2); driveR2 += att_a*(excR-driveR2); }
                    if(am3){ float excL = exL * md3->gain_nowL; float excR = exR * md3->gain_nowR; driveL3 += att_a*(excL-driveL3); driveR3 += att_a*(excR-driveR3); }
                    float xL_[4] = {driveL0, driveL1, driveL2, driveL3};
                    float xR_[4] = {driveR0, driveR1, driveR2, driveR3};
                    float32x4_t xL4 = vld1q_f32(xL_);
                    float32x4_t xR4 = vld1q_f32(xR_);
                
                    float32x4_t yL4 = jb_svf_bp_tick4(gL4, dL4, &s1L4, &s2L4, xL4);
                    float32x4_t yR4 = jb_svf_bp_tick4(gR4, dR4, &s1R4, &s2R4, xR4);
                
                    // Keep old state for inactive lanes; zero output for inactive lanes
                    s1L4 = vbslq_f32(activeMask, s1L4, vld1q_f32(s1L_));
                    s2L4 = vbslq_f32(activeMask, s2L4, vld1q_f32(s2L_));
                    s1R4 = vbslq_f32(activeMask, s1R4, vld1q_f32(s1R_));
                    s2R4 = vbslq_f32(activeMask, s2R4, vld1q_f32(s2R_));
                    yL4  = vbslq_f32(activeMask, yL4, vdupq_n_f32(0.f));
                    yR4  = vbslq_f32(activeMask, yR4, vdupq_n_f32(0.f));
                
                    float yLlane[4], yRlane[4], s1Llane[4], s2Llane[4], s1Rlane[4], s2Rlane[4];
                    vst1q_f32(yLlane, yL4); vst1q_f32(yRlane, yR4);
                    vst1q_f32(s1Llane, s1L4); vst1q_f32(s2Llane, s2L4);
                    vst1q_f32(s1Rlane, s1R4); vst1q_f32(s2Rlane, s2R4);
                
                    const float e = v->rel_env2;
                    if(am0){ md0->svfL.s1=s1Llane[0]; md0->svfL.s2=s2Llane[0]; md0->svfR.s1=s1Rlane[0]; md0->svfR.s2=s2Rlane[0]; md0->driveL=driveL0; md0->driveR=driveR0; float y0L=jb_kill_denorm(yLlane[0]); float y0R=jb_kill_denorm(yRlane[0]); md0->y_pre_lastL=y0L; md0->y_pre_lastR=y0R; b2OutL=jb_kill_denorm(b2OutL + (y0L*e)*bank_gain2); b2OutR=jb_kill_denorm(b2OutR + (y0R*e)*bank_gain2); }
                    if(am1){ md1->svfL.s1=s1Llane[1]; md1->svfL.s2=s2Llane[1]; md1->svfR.s1=s1Rlane[1]; md1->svfR.s2=s2Rlane[1]; md1->driveL=driveL1; md1->driveR=driveR1; float y1L=jb_kill_denorm(yLlane[1]); float y1R=jb_kill_denorm(yRlane[1]); md1->y_pre_lastL=y1L; md1->y_pre_lastR=y1R; b2OutL=jb_kill_denorm(b2OutL + (y1L*e)*bank_gain2); b2OutR=jb_kill_denorm(b2OutR + (y1R*e)*bank_gain2); }
                    if(am2){ md2->svfL.s1=s1Llane[2]; md2->svfL.s2=s2Llane[2]; md2->svfR.s1=s1Rlane[2]; md2->svfR.s2=s2Rlane[2]; md2->driveL=driveL2; md2->driveR=driveR2; float y2L=jb_kill_denorm(yLlane[2]); float y2R=jb_kill_denorm(yRlane[2]); md2->y_pre_lastL=y2L; md2->y_pre_lastR=y2R; b2OutL=jb_kill_denorm(b2OutL + (y2L*e)*bank_gain2); b2OutR=jb_kill_denorm(b2OutR + (y2R*e)*bank_gain2); }
                    if(am3){ md3->svfL.s1=s1Llane[3]; md3->svfL.s2=s2Llane[3]; md3->svfR.s1=s1Rlane[3]; md3->svfR.s2=s2Rlane[3]; md3->driveL=driveL3; md3->driveR=driveR3; float y3L=jb_kill_denorm(yLlane[3]); float y3R=jb_kill_denorm(yRlane[3]); md3->y_pre_lastL=y3L; md3->y_pre_lastR=y3R; b2OutL=jb_kill_denorm(b2OutL + (y3L*e)*bank_gain2); b2OutR=jb_kill_denorm(b2OutR + (y3R*e)*bank_gain2); }
                }
                // scalar smoothing
                for(; m < x->n_modes2; m++){ 
                
                    if(!base2[m].active) continue;
                    jb_mode_rt_t *md=&v->m2[m];
                    float gL = md->gain_nowL;
                    float gR = md->gain_nowR;
                    if ((fabsf(gL) + fabsf(gR)) <= 0.f || md->nyq_kill) continue;

                    float driveL = md->driveL;
                    float driveR = md->driveR;
                    const float att_a = 1.f;
                    float excL = exL * gL;
                    float excR = exR * gR;

                    driveL += att_a*(excL - driveL);
	                    float y_rawL = jb_svf_bp_tick(&md->svfL, driveL);
	                    y_rawL = jb_kill_denorm(y_rawL);

                    driveR += att_a*(excR - driveR);
	                    float y_rawR = jb_svf_bp_tick(&md->svfR, driveR);
	                    y_rawR = jb_kill_denorm(y_rawR);

                    md->driveL = driveL;
                    md->driveR = driveR;
	                    // Pre-master, pre-envelope signal snapshot (for meters / hit detection)
	                    md->y_pre_lastL = y_rawL;
	                    md->y_pre_lastR = y_rawR;

	                    // SUM into bank-2 voice output (no pan)
	                    float e2 = v->rel_env2;
	                    b2OutL = jb_kill_denorm(b2OutL + (y_rawL * e2) * bank_gain2);
	                    b2OutR = jb_kill_denorm(b2OutR + (y_rawR * e2) * bank_gain2);
                }
                #else
                for(int m=0;m<x->n_modes2;m++){ 
                
                    if(!base2[m].active) continue;
                    jb_mode_rt_t *md=&v->m2[m];
                    float gL = md->gain_nowL;
                    float gR = md->gain_nowR;
                    if ((fabsf(gL) + fabsf(gR)) <= 0.f || md->nyq_kill) continue;

                    float driveL = md->driveL;
                    float driveR = md->driveR;
                    const float att_a = 1.f;
                    float excL = exL * gL;
                    float excR = exR * gR;

                    driveL += att_a*(excL - driveL);
	                    float y_rawL = jb_svf_bp_tick(&md->svfL, driveL);
	                    y_rawL = jb_kill_denorm(y_rawL);

                    driveR += att_a*(excR - driveR);
	                    float y_rawR = jb_svf_bp_tick(&md->svfR, driveR);
	                    y_rawR = jb_kill_denorm(y_rawR);

                    md->driveL = driveL;
                    md->driveR = driveR;
	                    // Pre-master, pre-envelope signal snapshot (for meters / hit detection)
	                    md->y_pre_lastL = y_rawL;
	                    md->y_pre_lastR = y_rawR;

	                    // SUM into bank-2 voice output (no pan)
	                    float e2 = v->rel_env2;
	                    b2OutL = jb_kill_denorm(b2OutL + (y_rawL * e2) * bank_gain2);
	                    b2OutR = jb_kill_denorm(b2OutR + (y_rawR * e2) * bank_gain2);
                }
                #endif
            }
            // BANK 1 input: internal exciter
            exL = ex0L;
            exR = ex0R;

// -------- BANK 1 --------
            if (bank_gain1 > 0.f && v->rel_env > 0.f){
                #if JB_HAVE_NEON && JB_ENABLE_NEON
                int m=0;
                for(; m+3 < x->n_modes; m+=4){
                    // Early skip if all 4 inactive
                    jb_mode_rt_t *md0=&v->m[m+0];
                    jb_mode_rt_t *md1=&v->m[m+1];
                    jb_mode_rt_t *md2=&v->m[m+2];
                    jb_mode_rt_t *md3=&v->m[m+3];
                    uint32_t am0 = (base1[m+0].active && !md0->nyq_kill && (fabsf(md0->gain_nowL)+fabsf(md0->gain_nowR))>0.f) ? 0xFFFFFFFFu : 0u;
                    uint32_t am1 = (base1[m+1].active && !md1->nyq_kill && (fabsf(md1->gain_nowL)+fabsf(md1->gain_nowR))>0.f) ? 0xFFFFFFFFu : 0u;
                    uint32_t am2 = (base1[m+2].active && !md2->nyq_kill && (fabsf(md2->gain_nowL)+fabsf(md2->gain_nowR))>0.f) ? 0xFFFFFFFFu : 0u;
                    uint32_t am3 = (base1[m+3].active && !md3->nyq_kill && (fabsf(md3->gain_nowL)+fabsf(md3->gain_nowR))>0.f) ? 0xFFFFFFFFu : 0u;
                    if(!(am0|am1|am2|am3)) continue;
                    uint32x4_t activeMask = (uint32x4_t){am0, am1, am2, am3};
                
                    // Gather SVF params/states (L)
                    float gL_[4]  = {md0->svfL.g,  md1->svfL.g,  md2->svfL.g,  md3->svfL.g};
                    float dL_[4]  = {md0->svfL.d,  md1->svfL.d,  md2->svfL.d,  md3->svfL.d};
                    float s1L_[4] = {md0->svfL.s1, md1->svfL.s1, md2->svfL.s1, md3->svfL.s1};
                    float s2L_[4] = {md0->svfL.s2, md1->svfL.s2, md2->svfL.s2, md3->svfL.s2};
                    float32x4_t gL4  = vld1q_f32(gL_);
                    float32x4_t dL4  = vld1q_f32(dL_);
                    float32x4_t s1L4 = vld1q_f32(s1L_);
                    float32x4_t s2L4 = vld1q_f32(s2L_);
                
                    // Gather SVF params/states (R)
                    float gR_[4]  = {md0->svfR.g,  md1->svfR.g,  md2->svfR.g,  md3->svfR.g};
                    float dR_[4]  = {md0->svfR.d,  md1->svfR.d,  md2->svfR.d,  md3->svfR.d};
                    float s1R_[4] = {md0->svfR.s1, md1->svfR.s1, md2->svfR.s1, md3->svfR.s1};
                    float s2R_[4] = {md0->svfR.s2, md1->svfR.s2, md2->svfR.s2, md3->svfR.s2};
                    float32x4_t gR4  = vld1q_f32(gR_);
                    float32x4_t dR4  = vld1q_f32(dR_);
                    float32x4_t s1R4 = vld1q_f32(s1R_);
                    float32x4_t s2R4 = vld1q_f32(s2R_);
                
                    // Drive update + input vectors
                    float driveL0=md0->driveL, driveL1=md1->driveL, driveL2=md2->driveL, driveL3=md3->driveL;
                    float driveR0=md0->driveR, driveR1=md1->driveR, driveR2=md2->driveR, driveR3=md3->driveR;
                    const float att_a = 1.f;
                    if(am0){ float excL = exL * md0->gain_nowL; float excR = exR * md0->gain_nowR; driveL0 += att_a*(excL-driveL0); driveR0 += att_a*(excR-driveR0); }
                    if(am1){ float excL = exL * md1->gain_nowL; float excR = exR * md1->gain_nowR; driveL1 += att_a*(excL-driveL1); driveR1 += att_a*(excR-driveR1); }
                    if(am2){ float excL = exL * md2->gain_nowL; float excR = exR * md2->gain_nowR; driveL2 += att_a*(excL-driveL2); driveR2 += att_a*(excR-driveR2); }
                    if(am3){ float excL = exL * md3->gain_nowL; float excR = exR * md3->gain_nowR; driveL3 += att_a*(excL-driveL3); driveR3 += att_a*(excR-driveR3); }
                    float xL_[4] = {driveL0, driveL1, driveL2, driveL3};
                    float xR_[4] = {driveR0, driveR1, driveR2, driveR3};
                    float32x4_t xL4 = vld1q_f32(xL_);
                    float32x4_t xR4 = vld1q_f32(xR_);
                
                    float32x4_t yL4 = jb_svf_bp_tick4(gL4, dL4, &s1L4, &s2L4, xL4);
                    float32x4_t yR4 = jb_svf_bp_tick4(gR4, dR4, &s1R4, &s2R4, xR4);
                
                    // Keep old state for inactive lanes; zero output for inactive lanes
                    s1L4 = vbslq_f32(activeMask, s1L4, vld1q_f32(s1L_));
                    s2L4 = vbslq_f32(activeMask, s2L4, vld1q_f32(s2L_));
                    s1R4 = vbslq_f32(activeMask, s1R4, vld1q_f32(s1R_));
                    s2R4 = vbslq_f32(activeMask, s2R4, vld1q_f32(s2R_));
                    yL4  = vbslq_f32(activeMask, yL4, vdupq_n_f32(0.f));
                    yR4  = vbslq_f32(activeMask, yR4, vdupq_n_f32(0.f));
                
                    float yLlane[4], yRlane[4], s1Llane[4], s2Llane[4], s1Rlane[4], s2Rlane[4];
                    vst1q_f32(yLlane, yL4); vst1q_f32(yRlane, yR4);
                    vst1q_f32(s1Llane, s1L4); vst1q_f32(s2Llane, s2L4);
                    vst1q_f32(s1Rlane, s1R4); vst1q_f32(s2Rlane, s2R4);
                
                    const float e = v->rel_env;
                    if(am0){ md0->svfL.s1=s1Llane[0]; md0->svfL.s2=s2Llane[0]; md0->svfR.s1=s1Rlane[0]; md0->svfR.s2=s2Rlane[0]; md0->driveL=driveL0; md0->driveR=driveR0; float y0L=jb_kill_denorm(yLlane[0]); float y0R=jb_kill_denorm(yRlane[0]); md0->y_pre_lastL=y0L; md0->y_pre_lastR=y0R; b1OutL=jb_kill_denorm(b1OutL + (y0L*e)*bank_gain1); b1OutR=jb_kill_denorm(b1OutR + (y0R*e)*bank_gain1); }
                    if(am1){ md1->svfL.s1=s1Llane[1]; md1->svfL.s2=s2Llane[1]; md1->svfR.s1=s1Rlane[1]; md1->svfR.s2=s2Rlane[1]; md1->driveL=driveL1; md1->driveR=driveR1; float y1L=jb_kill_denorm(yLlane[1]); float y1R=jb_kill_denorm(yRlane[1]); md1->y_pre_lastL=y1L; md1->y_pre_lastR=y1R; b1OutL=jb_kill_denorm(b1OutL + (y1L*e)*bank_gain1); b1OutR=jb_kill_denorm(b1OutR + (y1R*e)*bank_gain1); }
                    if(am2){ md2->svfL.s1=s1Llane[2]; md2->svfL.s2=s2Llane[2]; md2->svfR.s1=s1Rlane[2]; md2->svfR.s2=s2Rlane[2]; md2->driveL=driveL2; md2->driveR=driveR2; float y2L=jb_kill_denorm(yLlane[2]); float y2R=jb_kill_denorm(yRlane[2]); md2->y_pre_lastL=y2L; md2->y_pre_lastR=y2R; b1OutL=jb_kill_denorm(b1OutL + (y2L*e)*bank_gain1); b1OutR=jb_kill_denorm(b1OutR + (y2R*e)*bank_gain1); }
                    if(am3){ md3->svfL.s1=s1Llane[3]; md3->svfL.s2=s2Llane[3]; md3->svfR.s1=s1Rlane[3]; md3->svfR.s2=s2Rlane[3]; md3->driveL=driveL3; md3->driveR=driveR3; float y3L=jb_kill_denorm(yLlane[3]); float y3R=jb_kill_denorm(yRlane[3]); md3->y_pre_lastL=y3L; md3->y_pre_lastR=y3R; b1OutL=jb_kill_denorm(b1OutL + (y3L*e)*bank_gain1); b1OutR=jb_kill_denorm(b1OutR + (y3R*e)*bank_gain1); }
                }
                // scalar smoothing
                for(; m < x->n_modes; m++){ 
                
                    if(!base1[m].active) continue;
                    jb_mode_rt_t *md=&v->m[m];
                    float gL = md->gain_nowL;
                    float gR = md->gain_nowR;
                    if ((fabsf(gL) + fabsf(gR)) <= 0.f || md->nyq_kill) continue;

                    float driveL = md->driveL;
                    float driveR = md->driveR;
                    const float att_a = 1.f;
                    float excL = exL * gL;
                    float excR = exR * gR;

                    driveL += att_a*(excL - driveL);
	                    float y_rawL = jb_svf_bp_tick(&md->svfL, driveL);
	                    y_rawL = jb_kill_denorm(y_rawL);

                    driveR += att_a*(excR - driveR);
	                    float y_rawR = jb_svf_bp_tick(&md->svfR, driveR);
	                    y_rawR = jb_kill_denorm(y_rawR);

                    md->driveL = driveL;
                    md->driveR = driveR;
	                    md->y_pre_lastL = y_rawL;
	                    md->y_pre_lastR = y_rawR;

	                    // SUM into bank-1 voice output (no pan)
	                    float e1 = v->rel_env;
	                    b1OutL = jb_kill_denorm(b1OutL + (y_rawL * e1) * bank_gain1);
	                    b1OutR = jb_kill_denorm(b1OutR + (y_rawR * e1) * bank_gain1);
                }
                #else
                for(int m=0;m<x->n_modes;m++){ 
                
                    if(!base1[m].active) continue;
                    jb_mode_rt_t *md=&v->m[m];
                    float gL = md->gain_nowL;
                    float gR = md->gain_nowR;
                    if ((fabsf(gL) + fabsf(gR)) <= 0.f || md->nyq_kill) continue;

                    float driveL = md->driveL;
                    float driveR = md->driveR;
                    const float att_a = 1.f;
                    float excL = exL * gL;
                    float excR = exR * gR;

                    driveL += att_a*(excL - driveL);
	                    float y_rawL = jb_svf_bp_tick(&md->svfL, driveL);
	                    y_rawL = jb_kill_denorm(y_rawL);

                    driveR += att_a*(excR - driveR);
	                    float y_rawR = jb_svf_bp_tick(&md->svfR, driveR);
	                    y_rawR = jb_kill_denorm(y_rawR);

                    md->driveL = driveL;
                    md->driveR = driveR;
	                    md->y_pre_lastL = y_rawL;
	                    md->y_pre_lastR = y_rawR;

	                    // SUM into bank-1 voice output (no pan)
	                    float e1 = v->rel_env;
	                    b1OutL = jb_kill_denorm(b1OutL + (y_rawL * e1) * bank_gain1);
	                    b1OutR = jb_kill_denorm(b1OutR + (y_rawR * e1) * bank_gain1);
                }
                #endif
            }
            // Final per-voice sum (pre-space)
                // apply per-bank release envelopes then accumulate into output buffers
                b1OutL *= v->rel_env;
                b1OutR *= v->rel_env;
                b2OutL *= v->rel_env2;
                b2OutR *= v->rel_env2;

				float vOutL = b1OutL + b2OutL;
				float vOutR = b1OutR + b2OutR;

				// --- voice output + safety watchdog ---
				// If anything goes unstable (NaN/INF or runaway magnitude), hard-reset this voice.
				if (!jb_isfinitef(vOutL) || !jb_isfinitef(vOutR) ||
				    fabsf(vOutL) > JB_PANIC_ABS_MAX || fabsf(vOutR) > JB_PANIC_ABS_MAX){
				    jb_voice_panic_reset(x, v);
				    vOutL = 0.f;
				    vOutR = 0.f;
				}

				outL[i] += vOutL;
				outR[i] += vOutR;
                        // Release / voice-steal envelopes (feedback system removed)
            if (v->steal_countdown > 0){
                // Hard voice-steal fade: force envelope to countdown ratio (linear)
                float fade = (float)v->steal_countdown / (float)JB_STEAL_FADE_SAMPLES;
                if (fade < 0.f) fade = 0.f;
                v->rel_env  = fade;
                v->rel_env2 = fade;
                v->steal_countdown--;

                if (v->steal_countdown <= 0){
                    v->rel_env = v->rel_env2 = 0.f;
                    v->state = V_IDLE;
                    v->state = V_IDLE;
                }
            } else if (v->state == V_RELEASE){
                v->rel_env  *= a_rel1_sample;
                v->rel_env2 *= a_rel2_sample;

                if (v->rel_env < 1e-6f)  v->rel_env  = 0.f;
                if (v->rel_env2 < 1e-6f) v->rel_env2 = 0.f;

                if (v->rel_env <= 0.f && v->rel_env2 <= 0.f){
                    v->state = V_IDLE;
                    v->state = V_IDLE;
                }
            } else {
                v->rel_env  = 1.f;
                v->rel_env2 = 1.f;
            }

} // end samples
    } // end voices


    // Final safety: never output NaN/INF (can destabilize audio drivers / cause "freezing").
    for (int i = 0; i < n; ++i){
        if (!jb_isfinitef(outL[i])) outL[i] = 0.f;
        if (!jb_isfinitef(outR[i])) outR[i] = 0.f;
    }

    // ---------- SPACE (global stereo room) ----------
    // Schroeder-style: 4 combs per channel -> 2 allpasses per channel.
    // Bypass: only when SPACE wetdry is *totally dry* (-1).
    // NOTE: decay=0 is allowed and simply yields a very short/zero-feedback room.
    const float wetdry = jb_clamp(x->space_wetdry, -1.f, 1.f);
    if (wetdry > -1.f){
        const float size01 = jb_clamp(x->space_size, 0.f, 1.f);
        const float decay01 = jb_clamp(x->space_decay, 0.f, 1.f);
        const float diff01 = jb_clamp(x->space_diffusion, 0.f, 1.f);
        const float damp01 = jb_clamp(x->space_damping, 0.f, 1.f);

        const float size_scale = 0.05f + (size01 * 0.95f);
        float comb_g = powf(decay01, 1.5f) * 0.98f; // optional curve -> natural "long tails" at end
        if (comb_g < 0.f) comb_g = 0.f;
        if (comb_g > 0.98f) comb_g = 0.98f;

        float ap_g = 0.2f + (diff01 * 0.5f);
        ap_g = jb_clamp(ap_g, 0.2f, 0.7f);

        float damp = jb_clamp(damp01 * 0.8f, 0.f, 0.8f);

        // Delay taps (samples), scaled by SIZE. Clamped to safe bounds.
        int comb_delay[JB_SPACE_NCOMB];
        for (int k = 0; k < JB_SPACE_NCOMB; ++k){
            int d = (int)floorf((float)jb_space_base_delay[k] * size_scale + 0.5f);
            if (d < 1) d = 1;
            if (d > (JB_SPACE_MAX_DELAY - 2)) d = (JB_SPACE_MAX_DELAY - 2);
            comb_delay[k] = d;
        }

        // Wet/dry mapping: -1=dry, +1=wet
        const float mix = 0.5f * (wetdry + 1.f);
        const float dry_w = 1.f - mix;

        for (int i = 0; i < n; ++i){
            const float dryL = outL[i];
            const float dryR = outR[i];

            // L combs: 0..3, R combs: 4..7
            float comb_sumL = 0.f;
            float comb_sumR = 0.f;
            for (int k = 0; k < JB_SPACE_NCOMB_CH; ++k){
                comb_sumL += jb_space_comb_tick(x->space_comb_buf[k], JB_SPACE_MAX_DELAY,
                                               &x->space_comb_w[k], comb_delay[k],
                                               dryL, comb_g, damp, &x->space_comb_lp[k]);
                int rk = k + JB_SPACE_NCOMB_CH;
                comb_sumR += jb_space_comb_tick(x->space_comb_buf[rk], JB_SPACE_MAX_DELAY,
                                               &x->space_comb_w[rk], comb_delay[rk],
                                               dryR, comb_g, damp, &x->space_comb_lp[rk]);
            }

            // Normalize comb sum
            comb_sumL *= (1.f / (float)JB_SPACE_NCOMB_CH);
            comb_sumR *= (1.f / (float)JB_SPACE_NCOMB_CH);

            // Allpass diffusion (2 per channel)
            float wetL = comb_sumL;
            float wetR = comb_sumR;
            wetL = jb_space_ap_tick(x->space_ap_buf[0], JB_SPACE_AP_MAX, &x->space_ap_w[0], jb_space_ap_delay[0], wetL, ap_g);
            wetL = jb_space_ap_tick(x->space_ap_buf[1], JB_SPACE_AP_MAX, &x->space_ap_w[1], jb_space_ap_delay[1], wetL, ap_g);
            wetR = jb_space_ap_tick(x->space_ap_buf[2], JB_SPACE_AP_MAX, &x->space_ap_w[2], jb_space_ap_delay[2], wetR, ap_g);
            wetR = jb_space_ap_tick(x->space_ap_buf[3], JB_SPACE_AP_MAX, &x->space_ap_w[3], jb_space_ap_delay[3], wetR, ap_g);

            outL[i] = dryL * dry_w + wetL * mix;
            outR[i] = dryR * dry_w + wetR * mix;
        }
    }

    // Output DC HP (post-sum)
    float a=x->hp_a; float x1L=x->hpL_x1, y1L=x->hpL_y1, x1R=x->hpR_x1, y1R=x->hpR_y1;
    for(int i=0;i<n;i++){
        float xl=outL[i], xr=outR[i];
        float yl=a*(y1L + xl - x1L);
        float yr=a*(y1R + xr - x1R);
        if(fabsf(yl)<1e-20f){ yl=0.f; }
        if(fabsf(yr)<1e-20f){ yr=0.f; }
        outL[i]=yl; outR[i]=yr; x1L=xl; y1L=yl; x1R=xr; y1R=yr;
    }
    x->hpL_x1=x1L; x->hpL_y1=y1L; x->hpR_x1=x1R; x->hpR_y1=y1R;

    return (w + 5);}

// ---------- base setters & messages ----------
static void juicy_bank_tilde_modes(t_juicy_bank_tilde *x, t_floatarg nf){
    int n = (int)nf;
    if (n < 1) n = 1;
    if (n > JB_MAX_MODES) n = JB_MAX_MODES;

    int *n_modes_p   = x->edit_bank ? &x->n_modes2      : &x->n_modes;
    int *active_p    = x->edit_bank ? &x->active_modes2 : &x->active_modes;
    int *edit_idx_p  = x->edit_bank ? &x->edit_idx2     : &x->edit_idx;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;

    *n_modes_p = n;
    if (*active_p > *n_modes_p) *active_p = *n_modes_p;
    if (*active_p < 0) *active_p = 0;

    if (*edit_idx_p >= *n_modes_p) *edit_idx_p = *n_modes_p - 1;
    if (*edit_idx_p < 0) *edit_idx_p = 0;

    for (int i = 0; i < *n_modes_p; i++){
        if (base[i].base_ratio <= 0.f) base[i].base_ratio = (float)(i+1);
    }
}
static void juicy_bank_tilde_active(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg onf){
    int *n_modes_p   = x->edit_bank ? &x->n_modes2 : &x->n_modes;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;

    int idx = (int)idxf - 1;
    if (idx < 0 || idx >= *n_modes_p) return;
    base[idx].active = (onf > 0.f) ? 1 : 0;
}

// INDIVIDUAL (per-mode via index)
static void juicy_bank_tilde_index(t_juicy_bank_tilde *x, t_floatarg f){
    int *n_modes_p   = x->edit_bank ? &x->n_modes2  : &x->n_modes;
    int *edit_idx_p  = x->edit_bank ? &x->edit_idx2 : &x->edit_idx;

    int idx = (int)f;
    if (idx < 1) idx = 1;
    if (idx > *n_modes_p) idx = *n_modes_p;
    *edit_idx_p = idx - 1;
}
static void juicy_bank_tilde_ratio_i(t_juicy_bank_tilde *x, t_floatarg r){
    int *n_modes_p   = x->edit_bank ? &x->n_modes2  : &x->n_modes;
    int *edit_idx_p  = x->edit_bank ? &x->edit_idx2 : &x->edit_idx;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;

    int i = *edit_idx_p;
    if (i < 0 || i >= *n_modes_p) return;

    if (base[i].keytrack){
        float v = (r <= 0.f) ? 0.01f : r;
        base[i].base_ratio = v;
    } else {
        float ui = r;
        if (ui < 0.f) ui = 0.f;
        if (ui > 10.f) ui = 10.f;
        base[i].base_ratio = 100.f * ui;
    }
}
static void juicy_bank_tilde_gain_i(t_juicy_bank_tilde *x, t_floatarg g){
    int *n_modes_p   = x->edit_bank ? &x->n_modes2  : &x->n_modes;
    int *edit_idx_p  = x->edit_bank ? &x->edit_idx2 : &x->edit_idx;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;

    int i = *edit_idx_p;
    if (i < 0 || i >= *n_modes_p) return;
    base[i].base_gain = jb_clamp(g, 0.f, 1.f);
}
static void juicy_bank_tilde_decay_i(t_juicy_bank_tilde *x, t_floatarg ms){
    int *n_modes_p   = x->edit_bank ? &x->n_modes2  : &x->n_modes;
    int *edit_idx_p  = x->edit_bank ? &x->edit_idx2 : &x->edit_idx;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;

    int i = *edit_idx_p;
    if (i < 0 || i >= *n_modes_p) return;
    base[i].base_decay_ms = (ms < 0.f) ? 0.f : ms;
}

// Per-mode lists
static void juicy_bank_tilde_freq(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;
    for(int i=0; i<argc && i<JB_MAX_MODES; i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i);
            base[i].base_ratio = (v <= 0.f) ? 0.01f : v;
        }
    }
}
static void juicy_bank_tilde_decays(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;
    for(int i=0; i<argc && i<JB_MAX_MODES; i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i);
            base[i].base_decay_ms = (v < 0.f) ? 0.f : v;
        }
    }
}
static void juicy_bank_tilde_amps(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;
    for(int i=0; i<argc && i<JB_MAX_MODES; i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i);
            base[i].base_gain = jb_clamp(v, 0.f, 1.f);
        }
    }
}

// BODY (Type-4 Caughey single-bell damping)
//
// Pd dials often output 0..1. For usability, each inlet accepts either:
//   • 0..1   = "knob" value mapped to a sensible physical range (log where needed)
//   • >1     = direct value (power-user)
//
// Inlets (per selected bank):
//   bell_freq  : peak frequency in Hz (omega_p = 2*pi*Hz)
//   bell_zeta  : peak damping ratio at the peak (zeta_p)
//   bell_npl   : left power  (n_pl > 0)
//   bell_npr   : right power (n_pr > 0)
//   bell_npm   : model parameter (n_pm > -2)
//
// The curve is then:
//   zeta(omega) = zeta_p * (2+n_pm) / ( (omega/omega_p)^(-n_pl) + n_pm + (omega/omega_p)^(n_pr) )

#ifndef JB_BELL_ZETA_MIN
#define JB_BELL_ZETA_MIN 1e-6f
#endif
#ifndef JB_BELL_ZETA_MAX
#define JB_BELL_ZETA_MAX 2e-2f
#endif
#ifndef JB_BELL_PEAK_HZ_MIN
#define JB_BELL_PEAK_HZ_MIN 50.f
#endif
#ifndef JB_BELL_PEAK_HZ_MAX
#define JB_BELL_PEAK_HZ_MAX 12000.f
#endif
#ifndef JB_BELL_NP_MIN
#define JB_BELL_NP_MIN 0.25f
#endif
#ifndef JB_BELL_NP_MAX
#define JB_BELL_NP_MAX 8.f
#endif
#ifndef JB_BELL_NPM_MIN
#define JB_BELL_NPM_MIN -1.5f
#endif
#ifndef JB_BELL_NPM_MAX
#define JB_BELL_NPM_MAX 20.f
#endif

static inline float jb_bell_map_norm_to_zeta(float u){
    u = jb_clamp(u, 0.f, 1.f);
    const float zmin = JB_BELL_ZETA_MIN;
    const float zmax = JB_BELL_ZETA_MAX;
    return expf(logf(zmin) + u * (logf(zmax) - logf(zmin)));
}
static inline float jb_bell_param_to_zeta(float f){
    if (!isfinite(f)) return 0.f;
    if (f <= 1.f) return jb_bell_map_norm_to_zeta(f);
    return jb_clamp(f, 0.f, 0.2f);
}

static inline float jb_bell_map_norm_to_hz(const t_juicy_bank_tilde *x, float u){
    u = jb_clamp(u, 0.f, 1.f);
    float sr = x->sr;
    float nyq = (sr > 1.f) ? (0.5f * sr) : 22050.f;
    float fmin = JB_BELL_PEAK_HZ_MIN;
    float fmax = JB_BELL_PEAK_HZ_MAX;
    if (fmax > 0.95f * nyq) fmax = 0.95f * nyq;
    if (fmax < fmin) fmax = fmin;
    return expf(logf(fmin) + u * (logf(fmax) - logf(fmin)));
}
static inline float jb_bell_param_to_hz(const t_juicy_bank_tilde *x, float f){
    if (!isfinite(f)) return JB_BELL_PEAK_HZ_MIN;
    float out = (f <= 1.f) ? jb_bell_map_norm_to_hz(x, f) : f;
    float sr = x->sr;
    float nyq = (sr > 1.f) ? (0.5f * sr) : 22050.f;
    if (out < 1.f) out = 1.f;
    if (out > 0.95f * nyq) out = 0.95f * nyq;
    return out;
}

static inline float jb_bell_map_norm_to_pow(float u){
    u = jb_clamp(u, 0.f, 1.f);
    const float pmin = JB_BELL_NP_MIN;
    const float pmax = JB_BELL_NP_MAX;
    return expf(logf(pmin) + u * (logf(pmax) - logf(pmin)));
}
static inline float jb_bell_param_to_pow(float f){
    if (!isfinite(f)) return 1.f;
    if (f <= 1.f) return jb_bell_map_norm_to_pow(f);
    if (f < JB_BELL_NP_MIN) f = JB_BELL_NP_MIN;
    if (f > 32.f) f = 32.f;
    return f;
}

static inline float jb_bell_map_norm_to_npm(float u){
    u = jb_clamp(u, 0.f, 1.f);
    return JB_BELL_NPM_MIN + u * (JB_BELL_NPM_MAX - JB_BELL_NPM_MIN);
}
static inline float jb_bell_param_to_npm(float f){
    if (!isfinite(f)) return 0.f;
    // Treat 0..1 as knob range; otherwise direct
    float out = (f >= 0.f && f <= 1.f) ? jb_bell_map_norm_to_npm(f) : f;
    if (out <= -1.99f) out = -1.99f;
    if (out > 100.f) out = 100.f;
    return out;
}

static void juicy_bank_tilde_bell_freq(t_juicy_bank_tilde *x, t_floatarg f){

    float v = jb_bell_param_to_hz(x, (float)f);
    int b = x->edit_bank ? 1 : 0;
    int d = x->edit_damper;
    if (d < 0) d = 0; else if (d >= JB_N_DAMPERS) d = JB_N_DAMPERS - 1;
    x->bell_peak_hz[b][d] = v;
}
static void juicy_bank_tilde_bell_zeta(t_juicy_bank_tilde *x, t_floatarg f){

    float v = jb_bell_param_to_zeta((float)f);
    float u = (isfinite(f) && f <= 1.f) ? jb_clamp((float)f, 0.f, 1.f) : -1.f;
    int b = x->edit_bank ? 1 : 0;
    int d = x->edit_damper;
    if (d < 0) d = 0; else if (d >= JB_N_DAMPERS) d = JB_N_DAMPERS - 1;
    x->bell_peak_zeta[b][d] = v;
    x->bell_peak_zeta_param[b][d] = (v <= 0.f) ? -1.f : u;
}
static void juicy_bank_tilde_bell_npl(t_juicy_bank_tilde *x, t_floatarg f){

    float v = jb_bell_param_to_pow((float)f);
    int b = x->edit_bank ? 1 : 0;
    int d = x->edit_damper;
    if (d < 0) d = 0; else if (d >= JB_N_DAMPERS) d = JB_N_DAMPERS - 1;
    x->bell_npl[b][d] = v;
}
static void juicy_bank_tilde_bell_npr(t_juicy_bank_tilde *x, t_floatarg f){

    float v = jb_bell_param_to_pow((float)f);
    int b = x->edit_bank ? 1 : 0;
    int d = x->edit_damper;
    if (d < 0) d = 0; else if (d >= JB_N_DAMPERS) d = JB_N_DAMPERS - 1;
    x->bell_npr[b][d] = v;
}
static void juicy_bank_tilde_bell_npm(t_juicy_bank_tilde *x, t_floatarg f){

    float v = jb_bell_param_to_npm((float)f);
    int b = x->edit_bank ? 1 : 0;
    int d = x->edit_damper;
    if (d < 0) d = 0; else if (d >= JB_N_DAMPERS) d = JB_N_DAMPERS - 1;
    x->bell_npm[b][d] = v;
}

static void juicy_bank_tilde_damper_sel(t_juicy_bank_tilde *x, t_floatarg f){
    // UI: 1..JB_N_DAMPERS
    int d = (int)floorf(((float)f) + 0.5f);
    if (d < 1) d = 1;
    if (d > JB_N_DAMPERS) d = JB_N_DAMPERS;
    x->edit_damper = d - 1;
}


static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){
    if (x->edit_bank) x->brightness2 = jb_clamp(f, -1.f, 1.f);
    else              x->brightness  = jb_clamp(f, -1.f, 1.f);
}
static void juicy_bank_tilde_density(t_juicy_bank_tilde *x, t_floatarg f){
    float v = (float)f;
    if (v < -1.f) v = -1.f;
    if (x->edit_bank) x->density_amt2 = v;
    else              x->density_amt  = v;
}

static void juicy_bank_tilde_density_pivot(t_juicy_bank_tilde *x){
    if (x->edit_bank) x->density_mode2 = DENSITY_PIVOT;
    else              x->density_mode  = DENSITY_PIVOT;
}
static void juicy_bank_tilde_density_individual(t_juicy_bank_tilde *x){
    if (x->edit_bank) x->density_mode2 = DENSITY_INDIV;
    else              x->density_mode  = DENSITY_INDIV;
}
static void juicy_bank_tilde_release(t_juicy_bank_tilde *x, t_floatarg f){
    if (x->edit_bank) x->release_amt2 = jb_clamp(f, 0.f, 1.f);
    else              x->release_amt  = jb_clamp(f, 0.f, 1.f);
}

// realism & misc

static void juicy_bank_tilde_micro_detune(t_juicy_bank_tilde *x, t_floatarg f){
    if (x->edit_bank) x->micro_detune2 = jb_clamp(f, 0.f, 1.f);
    else              x->micro_detune  = jb_clamp(f, 0.f, 1.f);
}
// --- Position / Pickup setters (1D, Elements-style) ---
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){
    float v = jb_clamp(f, 0.f, 1.f);
    if (x->edit_bank) x->excite_pos2 = v;
    else              x->excite_pos  = v;
}
static void juicy_bank_tilde_pickup(t_juicy_bank_tilde *x, t_floatarg f){
    float v = jb_clamp(f, 0.f, 1.f);
    if (x->edit_bank) x->pickup_pos2 = v;
    else              x->pickup_pos  = v;
}

static void juicy_bank_tilde_odd_even(t_juicy_bank_tilde *x, t_floatarg f){
    // Odd vs Even emphasis bias: -1..+1
    // -1 => silence even modes, 0 => neutral, +1 => silence odd modes
    float v = jb_clamp(f, -1.f, 1.f);
    if (x->edit_bank) x->odd_even_bias2 = v;
    else              x->odd_even_bias  = v;
}

// --- LFO + ADSR param setters (for modulation matrix) ---
static void juicy_bank_tilde_lfo_shape(t_juicy_bank_tilde *x, t_floatarg f){
    int s = (int)floorf(f + 0.5f);
    if (s < 1) s = 1;
    if (s > 5) s = 5;
    x->lfo_shape = (float)s;

    // write into the currently selected LFO slot
    int idx = (int)floorf(x->lfo_index + 0.5f) - 1;
    if (idx < 0) idx = 0;
    if (idx >= JB_N_LFO) idx = JB_N_LFO - 1;
    x->lfo_shape_v[idx] = (float)s;
}
static void juicy_bank_tilde_lfo_rate(t_juicy_bank_tilde *x, t_floatarg f){
    float r = jb_clamp(f, 0.f, 20.f);
    x->lfo_rate = r;

    // write into the currently selected LFO slot
    int idx = (int)floorf(x->lfo_index + 0.5f) - 1;
    if (idx < 0) idx = 0;
    if (idx >= JB_N_LFO) idx = JB_N_LFO - 1;
    x->lfo_rate_v[idx] = r;
}
static void juicy_bank_tilde_lfo_phase(t_juicy_bank_tilde *x, t_floatarg f){
    float p = f - floorf(f);
    if (p < 0.f) p += 1.f;
    x->lfo_phase = p;

    // write into the currently selected LFO slot
    int idx = (int)floorf(x->lfo_index + 0.5f) - 1;
    if (idx < 0) idx = 0;
    if (idx >= JB_N_LFO) idx = JB_N_LFO - 1;
    x->lfo_phase_v[idx] = p;
}
static void juicy_bank_tilde_lfo_mode(t_juicy_bank_tilde *x, t_floatarg f){
    int m = (int)floorf(f + 0.5f);
    if (m < 1) m = 1;
    if (m > 2) m = 2;
    x->lfo_mode = (float)m;

    // write into the currently selected LFO slot
    int idx = (int)floorf(x->lfo_index + 0.5f) - 1;
    if (idx < 0) idx = 0;
    if (idx >= JB_N_LFO) idx = JB_N_LFO - 1;
    x->lfo_mode_v[idx] = (float)m;
}

static void juicy_bank_tilde_lfo_index(t_juicy_bank_tilde *x, t_floatarg f){
    int idx = (int)floorf(f + 0.5f);
    if (idx < 1) idx = 1;
    if (idx > JB_N_LFO) idx = JB_N_LFO;
    x->lfo_index = (float)idx;

    // when switching LFO index, mirror that slot back to the scalar view
    int li = idx - 1;
    if (li < 0) li = 0;
    if (li >= JB_N_LFO) li = JB_N_LFO - 1;
    x->lfo_shape = x->lfo_shape_v[li];
    x->lfo_rate  = x->lfo_rate_v[li];
    x->lfo_phase = x->lfo_phase_v[li];
    x->lfo_amount = x->lfo_amt_v[li];
    x->lfo_mode  = x->lfo_mode_v[li];
}

static void juicy_bank_tilde_lfo_amount(t_juicy_bank_tilde *x, t_floatarg f){
    float a = jb_clamp(f, -1.f, 1.f);
    x->lfo_amount = a;

    int idx = (int)floorf(x->lfo_index + 0.5f) - 1;
    if (idx < 0) idx = 0;
    if (idx >= JB_N_LFO) idx = JB_N_LFO - 1;
    x->lfo_amt_v[idx] = a;
}

// --- target assignment helpers (symbols) ---
static inline int jb_target_is_none(t_symbol *s){
    return (!s || s == gensym("none"));
}
static inline int jb_target_taken(const t_juicy_bank_tilde *x, t_symbol *tgt, int exclude_lane){
    if (jb_target_is_none(tgt)) return 0;
    // lanes: 0=LFO1, 1=LFO2
    if (exclude_lane != 0 && x->lfo_target[0] == tgt) return 1;
    if (exclude_lane != 1 && x->lfo_target[1] == tgt) return 1;
    return 0;
}


static inline int jb_lfo_target_allowed(t_symbol *s){
    // Note: Called on control-rate message paths (not per-sample DSP),
    // so using gensym() here is fine.
    if (jb_target_is_none(s)) return 1;
    return (
        s == gensym("master_1") || s == gensym("master_2") ||
        s == gensym("pitch_1")  || s == gensym("pitch_2")  ||
        s == gensym("brightness_1") || s == gensym("brightness_2") ||
        s == gensym("partials_1")   || s == gensym("partials_2")   ||
        s == gensym("position")     || s == gensym("pickup")       ||
        s == gensym("position_1")   || s == gensym("position_2")   ||
        s == gensym("pickup_1")     || s == gensym("pickup_2")     ||
        s == gensym("imp_shape")    ||  /* maps to x->exc_imp_shape (0..1) */
        s == gensym("noise_timbre") ||  /* maps to x->exc_shape     (0..1) */
        s == gensym("lfo2_rate")    ||
        s == gensym("lfo2_amount")
    );
}

// Velocity-mapping target whitelist (separate lane from the existing mod-matrix velocity source).
static inline int jb_velmap_target_allowed(t_symbol *s){
    if (!s) return 0;
    if (jb_target_is_none(s)) return 1;

    // bell_z_damper{1..3}_{1..2}
    if (!strncmp(s->s_name, "bell_z_damper", 13)){
        int damper=0, bank=0;
        if (sscanf(s->s_name, "bell_z_damper%d_%d", &damper, &bank) == 2){
            return (damper>=1 && damper<=JB_N_DAMPERS && bank>=1 && bank<=2);
        }
        return 0;
    }

    // bank-suffixed targets
    if (!strcmp(s->s_name, "brightness_1") || !strcmp(s->s_name, "brightness_2")) return 1;
    if (!strcmp(s->s_name, "position_1")   || !strcmp(s->s_name, "position_2"))   return 1;
    if (!strcmp(s->s_name, "pickup_1")     || !strcmp(s->s_name, "pickup_2"))     return 1;
    if (!strcmp(s->s_name, "master_1")     || !strcmp(s->s_name, "master_2"))     return 1;

    // noise/exciter ADSR + timbre (no bank suffix)
    if (!strcmp(s->s_name, "adsr_attack"))  return 1;
    if (!strcmp(s->s_name, "adsr_decay"))   return 1;
    if (!strcmp(s->s_name, "adsr_release")) return 1;
    if (!strcmp(s->s_name, "imp_shape"))    return 1;
    if (!strcmp(s->s_name, "noise_timbre")) return 1;

    return 0;
}



static inline int jb_velmap_symbol_to_idx(const t_symbol *s){
    if (!s) return -1;

    // bell_z_damper{1..3}_{1..2}
    if (!strncmp(s->s_name, "bell_z_damper", 13)){
        int damper=0, bank=0;
        if (sscanf(s->s_name, "bell_z_damper%d_%d", &damper, &bank) == 2){
            if (damper >= 1 && damper <= 3 && (bank == 1 || bank == 2)){
                return (damper - 1) * 2 + (bank - 1); // 0..5
            }
        }
        return -1;
    }

    if (!strcmp(s->s_name, "brightness_1")) return JB_VEL_BRIGHTNESS_1;
    if (!strcmp(s->s_name, "brightness_2")) return JB_VEL_BRIGHTNESS_2;
    if (!strcmp(s->s_name, "position_1"))   return JB_VEL_POSITION_1;
    if (!strcmp(s->s_name, "position_2"))   return JB_VEL_POSITION_2;
    if (!strcmp(s->s_name, "pickup_1"))     return JB_VEL_PICKUP_1;
    if (!strcmp(s->s_name, "pickup_2"))     return JB_VEL_PICKUP_2;
    if (!strcmp(s->s_name, "master_1"))     return JB_VEL_MASTER_1;
    if (!strcmp(s->s_name, "master_2"))     return JB_VEL_MASTER_2;

    if (!strcmp(s->s_name, "adsr_attack"))  return JB_VEL_ADSR_ATTACK;
    if (!strcmp(s->s_name, "adsr_decay"))   return JB_VEL_ADSR_DECAY;
    if (!strcmp(s->s_name, "adsr_release")) return JB_VEL_ADSR_RELEASE;

    if (!strcmp(s->s_name, "imp_shape"))    return JB_VEL_IMP_SHAPE;
    if (!strcmp(s->s_name, "noise_timbre")) return JB_VEL_NOISE_TIMBRE;

    return -1;
}






static void juicy_bank_tilde_lfo1_target(t_juicy_bank_tilde *x, t_symbol *s){
    if (!s) return;
    if (!jb_lfo_target_allowed(s)){
        s = gensym("none");
    }
    if (jb_target_is_none(s)){
        x->lfo_target[0] = gensym("none");
        return;
    }
    if (jb_target_taken(x, s, 0)) return; // reject duplicate target
    x->lfo_target[0] = s;
}

static void juicy_bank_tilde_lfo2_target(t_juicy_bank_tilde *x, t_symbol *s){
    if (!s) return;
    if (!jb_lfo_target_allowed(s)){
        s = gensym("none");
    }
    if (jb_target_is_none(s)){
        x->lfo_target[1] = gensym("none");
        return;
    }
    if (jb_target_taken(x, s, 1)) return;
    x->lfo_target[1] = s;
}

static void juicy_bank_tilde_velmap_amount(t_juicy_bank_tilde *x, t_floatarg f){
    x->velmap_amount = jb_clamp((float)f, -1.f, 1.f);
}

static void juicy_bank_tilde_velmap_target(t_juicy_bank_tilde *x, t_symbol *s){
    if (!x || !s) return;
    if (!jb_velmap_target_allowed(s)){
        s = gensym("none");
    }
    if (jb_target_is_none(s)){
        // "none" clears all velocity-mapping toggles
        for (int i = 0; i < JB_VELMAP_N_TARGETS; ++i) x->velmap_on[i] = 0;
        x->velmap_target = gensym("none");
        return;
    }

    int idx = jb_velmap_symbol_to_idx(s);
    if (idx < 0 || idx >= JB_VELMAP_N_TARGETS) return;

    // Toggle behavior: sending the same target again flips it off.
    x->velmap_on[idx] = (uint8_t)(!x->velmap_on[idx]);
    x->velmap_target = s; // keep last-selected symbol for UI feedback/debug
}







// ---------- target inlet proxy (accepts bare selectors like 'brightness_1') ----------
static void jb_tgtproxy_set(jb_tgtproxy *p, t_symbol *tgt){
    if (!p || !p->owner || !tgt) return;
    switch(p->lane){
        case 0: juicy_bank_tilde_lfo1_target(p->owner, tgt); break;
        case 1: juicy_bank_tilde_lfo2_target(p->owner, tgt); break;
        case 2: juicy_bank_tilde_velmap_target(p->owner, tgt); break;
        default: break;
    }
}
static void jb_tgtproxy_float(jb_tgtproxy *p, t_floatarg f){ (void)p; (void)f; } // ignore floats
static void jb_tgtproxy_symbol(jb_tgtproxy *p, t_symbol *s){ jb_tgtproxy_set(p, s); }
static void jb_tgtproxy_list(jb_tgtproxy *p, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    if (argc < 1) return;
    if (argv[0].a_type == A_SYMBOL) jb_tgtproxy_set(p, atom_getsymbol(argv));
}
static void jb_tgtproxy_anything(jb_tgtproxy *p, t_symbol *s, int argc, t_atom *argv){
    if (!p) return;
    // If user sends "symbol foo" or "list foo", use first atom. Otherwise treat selector as the target.
    if ((s == &s_symbol || s == &s_list) && argc >= 1){
        if (argv[0].a_type == A_SYMBOL) jb_tgtproxy_set(p, atom_getsymbol(argv));
        return;
    }
    // Bare message box like "damper_1" comes through here with argc==0 and selector==gensym("damper_1")
    jb_tgtproxy_set(p, s);
}

static void jb_presetproxy_symbol(jb_presetproxy *p, t_symbol *s){
    if (!p || !p->owner) return;
    juicy_bank_tilde_preset_cmd(p->owner, s);
}

static void jb_presetproxy_anything(jb_presetproxy *p, t_symbol *s, int argc, t_atom *argv){
    if (!p || !p->owner) return;

    // Allow "symbol FORWARD" style too.
    if (s == &s_symbol && argc >= 1 && argv[0].a_type == A_SYMBOL){
        juicy_bank_tilde_preset_cmd(p->owner, atom_getsymbol(argv));
        return;
    }
    // Bare message box like [FORWARD( arrives here with selector "FORWARD".
    juicy_bank_tilde_preset_cmd(p->owner, s);
}


static void juicy_bank_tilde_odd_skew(t_juicy_bank_tilde *x, t_floatarg f){
    float v = jb_clamp(f, -1.f, 1.f);
    if (x->edit_bank) x->odd_skew2 = v;
    else              x->odd_skew  = v;
}

static void juicy_bank_tilde_even_skew(t_juicy_bank_tilde *x, t_floatarg f){
    float v = jb_clamp(f, -1.f, 1.f);
    if (x->edit_bank) x->even_skew2 = v;
    else              x->even_skew  = v;
}

static void juicy_bank_tilde_collision(t_juicy_bank_tilde *x, t_floatarg f){
    float v = jb_clamp(f, 0.f, 1.f);
    if (x->edit_bank) x->collision_amt2 = v;
    else              x->collision_amt  = v;
}

// ---------- SPACE (global room) ----------
static void juicy_bank_tilde_space_size(t_juicy_bank_tilde *x, t_floatarg f){
    x->space_size = jb_clamp(f, 0.f, 1.f);
}
static void juicy_bank_tilde_space_decay(t_juicy_bank_tilde *x, t_floatarg f){
    x->space_decay = jb_clamp(f, 0.f, 1.f);
}
static void juicy_bank_tilde_space_diffusion(t_juicy_bank_tilde *x, t_floatarg f){
    x->space_diffusion = jb_clamp(f, 0.f, 1.f);
}
static void juicy_bank_tilde_space_damping(t_juicy_bank_tilde *x, t_floatarg f){
    x->space_damping = jb_clamp(f, 0.f, 1.f);
}
static void juicy_bank_tilde_space_wetdry(t_juicy_bank_tilde *x, t_floatarg f){
    x->space_wetdry = jb_clamp(f, -1.f, 1.f);
}


// dispersion & seeds

static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){
    // Legacy name kept for backward compatibility.
    // This parameter is now QUANTIZE: 0..1, snaps ratios toward whole integers.
    float v = jb_clamp(f, 0.f, 1.f);
    float *disp_p = x->edit_bank ? &x->dispersion2 : &x->dispersion;
    *disp_p = v;
}

static void juicy_bank_tilde_quantize(t_juicy_bank_tilde *x, t_floatarg f){
    // Preferred name: quantize 0..1
    juicy_bank_tilde_dispersion(x, f);
}
static void juicy_bank_tilde_seed(t_juicy_bank_tilde *x, t_floatarg f){
    int *n_modes_p = x->edit_bank ? &x->n_modes2 : &x->n_modes;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;

    jb_rng_seed(&x->rng, (unsigned int)((int)f*2654435761u));
    for(int i=0; i<*n_modes_p; i++){
        base[i].disp_signature = (i==0) ? 0.f : jb_rng_bi(&x->rng);
        base[i].micro_sig      = (i==0) ? 0.f : jb_rng_bi(&x->rng);
    }
    if (x->edit_bank) x->dispersion_last2 = x->dispersion2;
    else              x->dispersion_last  = x->dispersion;
}
static void juicy_bank_tilde_dispersion_reroll(t_juicy_bank_tilde *x){
    int *n_modes_p = x->edit_bank ? &x->n_modes2 : &x->n_modes;
    jb_mode_base_t *base = x->edit_bank ? x->base2 : x->base;

    for(int i=0; i<*n_modes_p; i++){
        base[i].disp_signature = (i==0) ? 0.f : jb_rng_bi(&x->rng);
    }
    if (x->edit_bank) x->dispersion_last2 = -1.f;
    else              x->dispersion_last  = -1.f;

    // re-apply current dispersion value
    float disp = x->edit_bank ? x->dispersion2 : x->dispersion;
    juicy_bank_tilde_dispersion(x, disp);
}

// BEHAVIOR amounts

// Notes/poly (non-voice-addressed)
static void juicy_bank_tilde_note(t_juicy_bank_tilde *x, t_floatarg f0, t_floatarg vel){
    if (f0<=0.f){ f0=1.f; }
    jb_note_on(x, f0, vel);
}

static void juicy_bank_tilde_voices(t_juicy_bank_tilde *x, t_floatarg nf){
    (void)nf; x->max_voices = JB_MAX_VOICES;
    x->total_voices = JB_MAX_VOICES;
// fixed playable split
}

static void juicy_bank_tilde_note_midi(t_juicy_bank_tilde *x, t_floatarg midi, t_floatarg vel){
    // MIDI note -> Hz
    float f0 = (float)(440.0f * powf(2.0f, (midi - 69.0f) / 12.0f));
    if (f0<=0.f) f0 = 1.f;
    jb_note_on(x, f0, vel);
}
// basef0 reference (message)
static void juicy_bank_tilde_basef0(t_juicy_bank_tilde *x, t_floatarg f){ x->basef0_ref=(f<=0.f)?261.626f:f; }
static void juicy_bank_tilde_base_alias(t_juicy_bank_tilde *x, t_floatarg f){ juicy_bank_tilde_basef0(x,f); }

// reset/restart
static void juicy_bank_tilde_reset(t_juicy_bank_tilde *x){
    for(int v=0; v<JB_MAX_VOICES; ++v){
        x->v[v].state = V_IDLE; x->v[v].f0 = x->basef0_ref; x->v[v].vel = 0.f; x->v[v].energy=0.f; x->v[v].rel_env = 1.f; x->v[v].rel_env2 = 1.f;
        jb_exc_voice_reset_runtime(&x->v[v].exc);
        for (int b = 0; b < 2; ++b){
        }
        for(int i=0;i<JB_MAX_MODES;i++){
            x->v[v].disp_offset[i]=x->v[v].disp_target[i]=0.f;
            x->v[v].disp_offset2[i]=x->v[v].disp_target2[i]=0.f;
        }
    }
}
static void juicy_bank_tilde_restart(t_juicy_bank_tilde *x){ juicy_bank_tilde_reset(x); }

// preset recall helper: kill all active voice energy + reset internal exciter runtime
static void juicy_bank_tilde_preset_recall(t_juicy_bank_tilde *x){
    for(int v=0; v<JB_MAX_VOICES; ++v){
        x->v[v].state = V_IDLE;
        x->v[v].vel = 0.f;
        x->v[v].energy = 0.f;
        x->v[v].rel_env = 1.f;
        x->v[v].rel_env2 = 1.f;
        jb_exc_voice_reset_runtime(&x->v[v].exc);
        for (int b = 0; b < 2; ++b){
        }
    }
}

// ---------- dsp setup/free ----------
static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;
    float fc=8.f;  float RC=1.f/(2.f*M_PI*fc);  float dt=1.f/x->sr; x->hp_a=RC/(RC+dt);

    // sp layout: [outL, outR] (internal exciter; no external signal inlets)
    t_int argv[2 + 2 + 1];
    int a=0;
    argv[a++] = (t_int)x;
    for(int k=0;k<2;k++) argv[a++] = (t_int)(sp[k]->s_vec);
    argv[a++] = (int)(sp[0]->s_n);
    dsp_addv(juicy_bank_tilde_perform, a, argv);
}


// ---------- offline render (testing/regression) ----------
static void jb_render_free(t_juicy_bank_tilde *x){
    if (!x) return;
    if (x->render_bufL){ free(x->render_bufL); x->render_bufL = NULL; }
    if (x->render_bufR){ free(x->render_bufR); x->render_bufR = NULL; }
    x->render_len = 0;
    x->render_sr  = 0;
}

// Write 32-bit-float stereo WAV (little-endian).
static int jb_write_wav_f32_stereo(const char *path, const float *L, const float *R, int n, int sr){
    if (!path || !L || !R || n <= 0 || sr <= 0) return 0;
    FILE *fp = fopen(path, "wb");
    if (!fp) return 0;

    uint32_t data_bytes = (uint32_t)(n * 2 * (int)sizeof(float));
    uint32_t fmt_size   = 16u;
    uint32_t riff_size  = 4u + (8u + fmt_size) + (8u + data_bytes);

    // RIFF header
    fwrite("RIFF", 1, 4, fp);
    fwrite(&riff_size, 4, 1, fp);
    fwrite("WAVE", 1, 4, fp);

    // fmt chunk
    fwrite("fmt ", 1, 4, fp);
    fwrite(&fmt_size, 4, 1, fp);

    uint16_t audio_format   = 3u;   // IEEE float
    uint16_t num_channels   = 2u;
    uint32_t sample_rate    = (uint32_t)sr;
    uint16_t bits_per_samp  = 32u;
    uint16_t block_align    = (uint16_t)(num_channels * (bits_per_samp / 8u));
    uint32_t byte_rate      = sample_rate * (uint32_t)block_align;

    fwrite(&audio_format, 2, 1, fp);
    fwrite(&num_channels, 2, 1, fp);
    fwrite(&sample_rate, 4, 1, fp);
    fwrite(&byte_rate, 4, 1, fp);
    fwrite(&block_align, 2, 1, fp);
    fwrite(&bits_per_samp, 2, 1, fp);

    // data chunk
    fwrite("data", 1, 4, fp);
    fwrite(&data_bytes, 4, 1, fp);

    for (int i = 0; i < n; ++i){
        fwrite(&L[i], sizeof(float), 1, fp);
        fwrite(&R[i], sizeof(float), 1, fp);
    }

    fclose(fp);
    return 1;
}

// Message: render <seconds> [<path.wav>]
// - Always fills an internal buffer (x->render_bufL/R).
// - Optional path writes a stereo 32-bit-float WAV for quick A/B regression tests.
static void juicy_bank_tilde_render(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    if (!x) return;
    if (argc < 1){
        post("juicy_bank~: render <seconds> [path.wav]");
        return;
    }

    float seconds = atom_getfloat(argv);
    if (!isfinite(seconds) || seconds <= 0.f){
        post("juicy_bank~: render seconds must be > 0");
        return;
    }

    int sr = (int)((x->sr > 0.f) ? x->sr : sys_getsr());
    if (sr <= 0){
        post("juicy_bank~: render failed (unknown sample rate; turn DSP on once)");
        return;
    }

    // cap to avoid accidental huge allocations
    if (seconds > 120.f) seconds = 120.f;

    int total = (int)lrintf(seconds * (float)sr);
    if (total < 1) total = 1;

    jb_render_free(x);
    x->render_bufL = (float *)calloc((size_t)total, sizeof(float));
    x->render_bufR = (float *)calloc((size_t)total, sizeof(float));
    if (!x->render_bufL || !x->render_bufR){
        jb_render_free(x);
        post("juicy_bank~: render failed (allocation)");
        return;
    }
    x->render_len = total;
    x->render_sr  = sr;

    // Render by calling our perform routine in blocks.
    // NOTE: This advances internal voice/LFO state exactly like realtime DSP.
    const int block = 64;
    int offs = 0;
    while (offs < total){
        int n = (total - offs < block) ? (total - offs) : block;
        t_int w[5];
        w[1] = (t_int)x;
        w[2] = (t_int)(x->render_bufL + offs);
        w[3] = (t_int)(x->render_bufR + offs);
        w[4] = (t_int)n;
        (void)juicy_bank_tilde_perform(w);
        offs += n;
    }

    if (argc >= 2 && argv[1].a_type == A_SYMBOL){
        const char *path = atom_getsymbol(argv + 1)->s_name;
        if (jb_write_wav_f32_stereo(path, x->render_bufL, x->render_bufR, x->render_len, x->render_sr)){
            post("juicy_bank~: rendered %.3fs to %s", seconds, path);
        } else {
            post("juicy_bank~: render write failed: %s", path);
        }
    } else {
        post("juicy_bank~: rendered %.3fs (%d samples @ %d Hz) to internal buffer", seconds, x->render_len, x->render_sr);
    }
}

// Message: renderwrite <path.wav>
static void juicy_bank_tilde_renderwrite(t_juicy_bank_tilde *x, t_symbol *path){
    if (!x || !path) return;
    if (!x->render_bufL || !x->render_bufR || x->render_len <= 0 || x->render_sr <= 0){
        post("juicy_bank~: renderwrite: no render buffer (run 'render <seconds>' first)");
        return;
    }
    const char *p = path->s_name;
    if (jb_write_wav_f32_stereo(p, x->render_bufL, x->render_bufR, x->render_len, x->render_sr)){
        post("juicy_bank~: wrote %s", p);
    } else {
        post("juicy_bank~: renderwrite failed: %s", p);
    }
}

// Message: renderclear
static void juicy_bank_tilde_renderclear(t_juicy_bank_tilde *x){
    jb_render_free(x);
    post("juicy_bank~: render buffer cleared");
}

static void juicy_bank_tilde_free(t_juicy_bank_tilde *x){
    inlet_free(x->in_release);

    inlet_free(x->in_bell_peak_hz); inlet_free(x->in_bell_peak_zeta); inlet_free(x->in_bell_npl); inlet_free(x->in_bell_npr); inlet_free(x->in_bell_npm); inlet_free(x->in_damper_sel); inlet_free(x->in_brightness); inlet_free(x->in_density);
    inlet_free(x->in_warp); inlet_free(x->in_dispersion);
    inlet_free(x->in_odd_skew);
    inlet_free(x->in_even_skew);
    inlet_free(x->in_collision);

            inlet_free(x->in_stretch);
    inlet_free(x->in_position);
    inlet_free(x->in_pickup);    inlet_free(x->in_odd_even);
inlet_free(x->in_partials); // free 'partials' inlet
inlet_free(x->in_index); inlet_free(x->in_ratio); inlet_free(x->in_gain);
    inlet_free(x->in_decay);

    // Internal exciter inlets (Fusion STEP 1)
    if (x->in_exc_fader) inlet_free(x->in_exc_fader);
    if (x->in_exc_attack) inlet_free(x->in_exc_attack);
    if (x->in_exc_attack_curve) inlet_free(x->in_exc_attack_curve);
    if (x->in_exc_decay) inlet_free(x->in_exc_decay);
    if (x->in_exc_decay_curve) inlet_free(x->in_exc_decay_curve);
    if (x->in_exc_sustain) inlet_free(x->in_exc_sustain);
    if (x->in_exc_release) inlet_free(x->in_exc_release);
    if (x->in_exc_release_curve) inlet_free(x->in_exc_release_curve);
    if (x->in_exc_imp_shape) inlet_free(x->in_exc_imp_shape);
    if (x->in_exc_shape) inlet_free(x->in_exc_shape);

    // MOD SECTION inlets
    if (x->in_lfo_index) inlet_free(x->in_lfo_index);
    if (x->in_lfo_shape) inlet_free(x->in_lfo_shape);
    if (x->in_lfo_rate) inlet_free(x->in_lfo_rate);
    if (x->in_lfo_phase) inlet_free(x->in_lfo_phase);
    if (x->in_lfo_mode)  inlet_free(x->in_lfo_mode);
    if (x->in_lfo_amount) inlet_free(x->in_lfo_amount);
    if (x->in_lfo1_target) inlet_free(x->in_lfo1_target);
    if (x->in_lfo2_target) inlet_free(x->in_lfo2_target);

    // Target proxies
    if (x->tgtproxy_lfo1) { pd_free((t_pd *)x->tgtproxy_lfo1); x->tgtproxy_lfo1 = 0; }
    if (x->tgtproxy_lfo2) { pd_free((t_pd *)x->tgtproxy_lfo2); x->tgtproxy_lfo2 = 0; }
    if (x->tgtproxy_velmap) { pd_free((t_pd *)x->tgtproxy_velmap); x->tgtproxy_velmap = 0; }


    // Offline render buffer
    jb_render_free(x);

    outlet_free(x->outL); outlet_free(x->outR);

}

// ---------- defaults helper ----------
static void jb_apply_default_saw_bank(t_juicy_bank_tilde *x, int bank){
    // bank: 0 = bank1 fields, 1 = bank2 fields
    int *n_modes_p   = bank ? &x->n_modes2      : &x->n_modes;
    int *active_p    = bank ? &x->active_modes2 : &x->active_modes;
    int *edit_idx_p  = bank ? &x->edit_idx2     : &x->edit_idx;
    jb_mode_base_t *base = bank ? x->base2 : x->base;

    *n_modes_p  = JB_MAX_MODES;
    *active_p   = JB_MAX_MODES;
    *edit_idx_p = 0;

    for(int i=0;i<JB_MAX_MODES;i++){
        base[i].active = 1;
        base[i].base_ratio = (float)(i+1);
        base[i].base_decay_ms = 1000.f;   // 1 second
        // Flat per-mode gain by default (brightness defines the spectral slope)
        base[i].base_gain = 1.0f;        base[i].keytrack = 1;
        base[i].disp_signature = 0.f;
        base[i].micro_sig      = 0.f;
    }

    // sensible body defaults (per bank)
    if (!bank){
        // Longer, more playable defaults (these are direct zetas, not 0..1 knobs)
        x->brightness = 0.f;
        for (int d = 0; d < JB_N_DAMPERS; ++d){
            x->bell_peak_hz[0][d]   = 3000.f;
            x->bell_peak_zeta[0][d] = (d == 0) ? 0.00035f : 0.f;
        x->bell_peak_zeta_param[0][d] = (d == 0) ? 0.5915f : -1.f; // damper 1 active, others off by default
            x->bell_peak_zeta_param[0][d] = (d == 0) ? 0.5915f : -1.f;
            x->bell_npl[0][d]       = 0.7f;
            x->bell_npr[0][d]       = 2.5f;
            x->bell_npm[0][d]       = 0.f;
        }
        x->density_amt = 0.f; x->density_mode = DENSITY_PIVOT;
        x->dispersion = 0.f; x->dispersion_last = -1.f;
x->release_amt = 1.f;
    } else {
        x->brightness2 = 0.f;
        for (int d = 0; d < JB_N_DAMPERS; ++d){
            x->bell_peak_hz[1][d]   = 3000.f;
            x->bell_peak_zeta[1][d] = (d == 0) ? 0.00035f : 0.f; // damper 1 active, others off by default
            x->bell_peak_zeta_param[1][d] = (d == 0) ? 0.5915f : -1.f;
            x->bell_npl[1][d]       = 0.7f;
            x->bell_npr[1][d]       = 2.5f;
            x->bell_npm[1][d]       = 0.f;
        }
        x->density_amt2 = 0.f; x->density_mode2 = DENSITY_PIVOT;
        x->dispersion2 = 0.f; x->dispersion_last2 = -1.f;
x->release_amt2 = 1.f;
    }
}

static void jb_apply_default_saw(t_juicy_bank_tilde *x){
    jb_apply_default_saw_bank(x, 0);
}

// ---------- new() ----------
static void *juicy_bank_tilde_new(void){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)pd_new(juicy_bank_tilde_class);
    x->sr = sys_getsr(); if(x->sr<=0) x->sr=48000;

    // --- Startup spec (32 modes, real saw amplitude 1/n) ---
    jb_apply_default_saw(x);


// preset system init (memory-only)
x->preset_mode = JB_PRESET_MODE_NORMAL;
x->preset_cursor = 0;
x->preset_slot_sel = 0;
memset(x->preset_edit_name, 0, sizeof(x->preset_edit_name));
for (int pi = 0; pi < JB_PRESET_SLOTS; ++pi){
    x->presets[pi].used = 0;
    memset(x->presets[pi].name, 0, sizeof(x->presets[pi].name));
}

    // body defaults
    x->brightness=0.f; x->density_amt=0.f; x->density_mode=DENSITY_PIVOT;
    x->dispersion=0.f; x->dispersion_last=-1.f;

    // Type-4 bell damping defaults (stackable 3x; damper 1 active, others off)
    for (int d = 0; d < JB_N_DAMPERS; ++d){
        x->bell_peak_hz[0][d]   = 3000.f;
        x->bell_peak_zeta[0][d] = (d == 0) ? 0.00035f : 0.f;
        x->bell_peak_zeta_param[0][d] = (d == 0) ? 0.5915f : -1.f;
        x->bell_npl[0][d]       = 0.7f;
        x->bell_npr[0][d]       = 2.5f;
        x->bell_npm[0][d]       = 0.f;
    }

x->odd_skew = 0.f;
    x->even_skew = 0.f;
    x->collision_amt=0.f;

    // Stretch default
    x->stretch = 0.f;

        x->warp = 0.f;
// realism defaults
    x->excite_pos=0.33f; x->pickup_pos=0.33f;
x->odd_even_bias = 0.f; x->odd_even_bias2 = 0.f;
// bank 2 defaults: start as a functional copy of bank 1 (bank 2 is still silent by master=0)
x->n_modes2 = x->n_modes;
x->active_modes2 = x->active_modes;
x->edit_idx2 = x->edit_idx;
memcpy(x->base2, x->base, sizeof(x->base));

x->release_amt2   = x->release_amt;
x->stretch2       = x->stretch;
x->warp2          = x->warp;

for (int d = 0; d < JB_N_DAMPERS; ++d){
    x->bell_peak_hz[1][d]   = x->bell_peak_hz[0][d];
    x->bell_peak_zeta[1][d] = x->bell_peak_zeta[0][d];
    x->bell_npl[1][d]       = x->bell_npl[0][d];
    x->bell_npr[1][d]       = x->bell_npr[0][d];
    x->bell_npm[1][d]       = x->bell_npm[0][d];
}
x->brightness2    = x->brightness;

x->density_amt2   = x->density_amt;
x->density_mode2  = x->density_mode;

x->dispersion2      = x->dispersion;
x->dispersion_last2 = x->dispersion_last;

x->odd_skew2      = x->odd_skew;
    x->even_skew2     = x->even_skew;
    x->collision_amt2 = x->collision_amt;


x->micro_detune2  = x->micro_detune;


x->excite_pos2    = x->excite_pos;
    x->pickup_pos2   = x->pickup_pos;

    // LFO + ADSR defaults
    x->lfo_shape = 1.f;   // default: shape 1 (for currently selected LFO)
    x->lfo_rate  = 1.f;   // 1 Hz
    x->lfo_phase = 0.f;   // start at phase 0
    x->lfo_mode  = 1.f;   // free by default
    x->lfo_index = 1.f;   // LFO 1 selected by default

    // Internal exciter defaults (shared across BOTH banks)
    x->exc_fader = 0.f;

    x->exc_attack_ms     = 5.f;
    x->exc_attack_curve  = 0.f;
    x->exc_decay_ms      = 600.f;
    x->exc_decay_curve   = 0.f;
    x->exc_sustain       = 0.5f;
    x->exc_release_ms    = 400.f;
    x->exc_release_curve = 0.f;

    x->exc_density    = 1.f;  // Pressure (AGC target 0..1 -> 0..0.98)
    x->exc_imp_shape  = 0.5f;  // Impulse-only Shape (old shape logic)
    x->exc_shape      = 0.5f;  // Noise Color (slope EQ)
    // Feedback-loop AGC defaults (0..1 mapped to time constants in DSP)
    x->noise_scale  = 1.f;
    x->impulse_scale = 1.f;

    // Offline render buffer
    x->render_bufL = x->render_bufR = NULL;
    x->render_len = 0;
    x->render_sr  = 0;
// initialise per-LFO parameter and runtime state
    for (int li = 0; li < JB_N_LFO; ++li){
        x->lfo_shape_v[li]      = 1.f;
        x->lfo_rate_v[li]       = 1.f;
        x->lfo_phase_v[li]      = 0.f;
        x->lfo_mode_v[li]       = 1.f;
        x->lfo_phase_state[li]  = 0.f;
        x->lfo_val[li]          = 0.f;
        x->lfo_snh[li]          = 0.f;
    }

    // default new mod-lane scaffolding
    for (int li = 0; li < JB_N_LFO; ++li){
        x->lfo_amt_v[li] = 0.f;
        x->lfo_amt_eff[li] = 0.f;
        x->lfo_target[li] = gensym("none");
    }
    x->lfo_amount = 0.f;

    // velocity mapping lane defaults
    x->velmap_amount = 0.f;
    x->velmap_target = gensym("none");
    for (int i = 0; i < JB_VELMAP_N_TARGETS; ++i) x->velmap_on[i] = 0;


    // clear modulation matrix (bank 1 + bank 2)
    for(int i=0;i<JB_N_MODSRC;i++)
        for(int j=0;j<JB_N_MODTGT;j++){
            x->mod_matrix[i][j]  = 0.f;
            x->mod_matrix2[i][j] = 0.f;
        }
    x->basef0_ref=261.626f; // C4

    x->max_voices = JB_MAX_VOICES;
    x->total_voices = JB_MAX_VOICES;
    for(int v=0; v<JB_MAX_VOICES; ++v){
        x->v[v].state=V_IDLE; x->v[v].f0=x->basef0_ref; x->v[v].vel=0.f; x->v[v].energy=0.f; x->v[v].rel_env=0.f; x->v[v].rel_env2=0.f;
        for(int i=0;i<JB_MAX_MODES;i++){
            x->v[v].disp_offset[i]=x->v[v].disp_target[i]=0.f;
            x->v[v].disp_offset2[i]=x->v[v].disp_target2[i]=0.f;
        }
        // Internal exciter init (Fusion STEP 1)
        jb_exc_voice_init(&x->v[v].exc, x->sr, 0xC0FFEEull + 1337ull*(unsigned long long)(v*2));

    }

    jb_rng_seed(&x->rng, 0xC0FFEEu);
    x->hp_a=0.f; x->hpL_x1=x->hpL_y1=x->hpR_x1=x->hpR_y1=0.f;

    // Two-bank scaffolding (STEP 1): bank 1 selected; bank 2 silent by default
    x->edit_bank = 0;
    x->edit_damper = 0;
    x->bank_master[0] = 1.f;
    x->bank_master[1] = 0.f;
    x->bank_semitone[0] = 0;
    x->bank_semitone[1] = 0;
    x->bank_octave[0] = 0;
    x->bank_octave[1] = 0;
    x->bank_tune_cents[0] = 0.f;
    x->bank_tune_cents[1] = 0.f;

    // INLETS (Signal → Behavior → Body → Individual)
    // Signal:
    // per-voice pairs

    // Behavior (reduced)
    x->in_release    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("release")); // release 0..1

    // Body (order: bell_freq, bell_zeta, bell_npl, bell_npr, bell_npm, damper_sel, brightness, density, stretch, warp, quantize, odd_skew, even_skew, collision)
    x->in_bell_peak_hz   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("bell_freq"));
    x->in_bell_peak_zeta = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("bell_zeta"));
    x->in_bell_npl       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("bell_npl"));
    x->in_bell_npr       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("bell_npr"));
    x->in_bell_npm       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("bell_npm"));
    x->in_damper_sel    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damper_sel")); // 1..3
    x->in_brightness = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("brightness"));
    x->in_density    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("density"));
    x->in_stretch    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("stretch"));
    x->in_warp       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("warp"));
    x->in_dispersion = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("quantize"));
    x->in_odd_skew   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("odd_skew"));
    x->in_even_skew  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("even_skew"));
    x->in_collision  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("collision"));
    // Spatial coupling controls (Mutable-style 1D position/pickup)
    x->in_position   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("position"));    // excitation position 0..1
    x->in_pickup     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("pickup"));      // pickup position 0..1
    x->in_odd_even      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("odd_even"));      // -1..+1 (odd vs even emphasis)

// Individual

    // SPACE defaults (global room)
    x->space_size = 0.25f;
    x->space_decay = 0.35f;
    x->space_diffusion = 0.6f;
    x->space_damping = 0.25f;
    x->space_wetdry = -0.3f; // -1..+1 : -1=dry, +1=wet (default matches old mix≈0.35)

    for (int k = 0; k < JB_SPACE_NCOMB; ++k){
        x->space_comb_w[k] = 0;
        x->space_comb_lp[k] = 0.f;
        for (int n = 0; n < JB_SPACE_MAX_DELAY; ++n){
            x->space_comb_buf[k][n] = 0.f;
        }
    }
    for (int k = 0; k < JB_SPACE_NAP; ++k){
        x->space_ap_w[k] = 0;
        for (int n = 0; n < JB_SPACE_AP_MAX; ++n){
            x->space_ap_buf[k][n] = 0.f;
        }
    }

    x->in_partials   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("partials"));
    x->in_master     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("master"));
    x->in_octave     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("octave"));
    x->in_semitone   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("semitone"));
    x->in_tune       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("tune"));
    x->in_bank       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("bank"));
    x->in_space_size      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("space_size"));
    x->in_space_decay     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("space_decay"));
    x->in_space_diffusion = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("space_diffusion"));
    x->in_space_damping   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("space_damping"));
    x->in_space_wetdry   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("space_wetdry"));
    x->in_index      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("index"));
    x->in_ratio      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("ratio"));
    x->in_gain       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("gain"));
    x->in_decay      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("decay"));   // alias of decay
// Internal EXCITER params (12 inlets, left->right like juicy_exciter~ v2.3)
    // NOTE: These are simple float inlets bound directly to x->exc_* fields.
    x->in_exc_fader         = floatinlet_new(&x->x_obj, &x->exc_fader);

    x->in_exc_attack        = floatinlet_new(&x->x_obj, &x->exc_attack_ms);
    x->in_exc_attack_curve  = floatinlet_new(&x->x_obj, &x->exc_attack_curve);

    x->in_exc_decay         = floatinlet_new(&x->x_obj, &x->exc_decay_ms);
    x->in_exc_decay_curve   = floatinlet_new(&x->x_obj, &x->exc_decay_curve);

    x->in_exc_sustain       = floatinlet_new(&x->x_obj, &x->exc_sustain);

    x->in_exc_release       = floatinlet_new(&x->x_obj, &x->exc_release_ms);
    x->in_exc_release_curve = floatinlet_new(&x->x_obj, &x->exc_release_curve);

    // feedback-adjacent parameter removed:
    // "exc_density" (exc_density) inlet used to drive feedback AGC target.
    x->in_exc_imp_shape    = floatinlet_new(&x->x_obj, &x->exc_imp_shape);
    x->in_exc_shape         = floatinlet_new(&x->x_obj, &x->exc_shape);

    // Feedback AGC inlets (placed AFTER timbre/color, BEFORE LFO index)

    // LFO inlets
    // --- MOD SECTION (starts after exciter controls) ---
    x->in_lfo_index  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_index"));
    x->in_lfo_shape  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_shape"));
    x->in_lfo_rate   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_rate"));
    x->in_lfo_phase  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_phase"));
    x->in_lfo_mode   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_mode"));
    x->in_lfo_amount = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_amount"));

    // Target selector inlets must accept bare selectors (e.g. a message box containing "damper_1"),
    // so we route them through proxies that implement an 'anything' handler.
    x->tgtproxy_lfo1 = (jb_tgtproxy *)pd_new(jb_tgtproxy_class);
    x->tgtproxy_lfo2 = (jb_tgtproxy *)pd_new(jb_tgtproxy_class);
    x->tgtproxy_velmap = (jb_tgtproxy *)pd_new(jb_tgtproxy_class);

    if (x->tgtproxy_lfo1){ x->tgtproxy_lfo1->owner = x; x->tgtproxy_lfo1->lane = 0; }
    if (x->tgtproxy_lfo2){ x->tgtproxy_lfo2->owner = x; x->tgtproxy_lfo2->lane = 1; }
    if (x->tgtproxy_velmap){ x->tgtproxy_velmap->owner = x; x->tgtproxy_velmap->lane = 2; }

    x->in_lfo1_target = inlet_new(&x->x_obj, x->tgtproxy_lfo1 ? &x->tgtproxy_lfo1->p_pd : &x->x_obj.ob_pd, 0, 0);
    x->in_lfo2_target = inlet_new(&x->x_obj, x->tgtproxy_lfo2 ? &x->tgtproxy_lfo2->p_pd : &x->x_obj.ob_pd, 0, 0);

    // Velocity mapping inlets (placed after LFO2 target)
    x->in_velmap_amount = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("velmap_amount"));
    x->in_velmap_target = inlet_new(&x->x_obj, x->tgtproxy_velmap ? &x->tgtproxy_velmap->p_pd : &x->x_obj.ob_pd, 0, 0);
// Preset system inlets (placed after velocity mapping)
    x->presetproxy = (jb_presetproxy*)pd_new(jb_presetproxy_class);
    if (x->presetproxy) x->presetproxy->owner = x;
    // Use a proxy so bare message boxes like [FORWARD( work (they arrive as 'anything' selectors).
    x->in_preset_cmd  = inlet_new(&x->x_obj, x->presetproxy ? &x->presetproxy->p_pd : &x->x_obj.ob_pd, 0, 0);
    x->in_preset_char = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("preset_char"));






    // Outs
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);
// checkpoint revert init

    
x->out_preset_f = outlet_new(&x->x_obj, &s_float);  // preset mode as float (0/1/2)
    x->out_preset   = outlet_new(&x->x_obj, &s_symbol); // preset feedback outlet (symbols/lists)
x->out_index   = outlet_new(&x->x_obj, &s_float); // 1-based index reporter
    return (void *)x;
}



static void juicy_bank_tilde_INIT(t_juicy_bank_tilde *x);
// -------------------- PRESET HELPERS --------------------
static inline char jb_preset_char_from_index(int idx){
    // idx: 1..64
    // 1 = space
    // 2..27 = A..Z
    // 28..53 = a..z
    // 54..63 = 0..9
    // 64 = '_' (extra)
    if (idx <= 1) return ' ';
    if (idx >= 2 && idx <= 27) return (char)('A' + (idx - 2));
    if (idx >= 28 && idx <= 53) return (char)('a' + (idx - 28));
    if (idx >= 54 && idx <= 63) return (char)('0' + (idx - 54));
    return '_';
}

static inline void jb_preset_trim_name(char *s){
    int n = (int)strlen(s);
    while (n > 0 && s[n-1] == ' '){ s[n-1] = '\0'; --n; }
}

static void jb_preset_emit_ui(t_juicy_bank_tilde *x){
    if (!x || !x->out_preset) return;

    t_atom a[2];

    // mode
    SETFLOAT(&a[0], (t_float)x->preset_mode);
    outlet_anything(x->out_preset, gensym("preset_mode"), 1, a);

    if (x->out_preset_f) outlet_float(x->out_preset_f, (t_float)x->preset_mode);

    // slot (1-based)
    SETFLOAT(&a[0], (t_float)(x->preset_slot_sel + 1));
    outlet_anything(x->out_preset, gensym("preset_slot"), 1, a);

    // name (full string)
    SETSYMBOL(&a[0], gensym(x->preset_edit_name));
    outlet_anything(x->out_preset, gensym("preset_name"), 1, a);

    // cursor (useful in naming mode)
    SETFLOAT(&a[0], (t_float)x->preset_cursor);
    outlet_anything(x->out_preset, gensym("edit_cursor"), 1, a);
}

static int jb_preset_find_next_used(const t_juicy_bank_tilde *x, int start, int dir){
    // start: 0..JB_PRESET_SLOTS-1, dir: +1 or -1
    int i = start;
    for (int k = 0; k < JB_PRESET_SLOTS; ++k){
        i += dir;
        if (i < 0) i = JB_PRESET_SLOTS - 1;
        if (i >= JB_PRESET_SLOTS) i = 0;
        if (x->presets[i].used) return i;
    }
    return -1;
}

static void jb_preset_snapshot(const t_juicy_bank_tilde *x, jb_preset_t *p){
    if (!p) return;
    // NOTE: keep snapshot fields aligned with synth params (memory-only bank)
    p->bank_master[0] = x->bank_master[0];
    p->bank_master[1] = x->bank_master[1];
    p->bank_semitone[0] = x->bank_semitone[0];
    p->bank_semitone[1] = x->bank_semitone[1];
    p->bank_octave[0] = x->bank_octave[0];
    p->bank_octave[1] = x->bank_octave[1];
    p->bank_tune_cents[0] = x->bank_tune_cents[0];
    p->bank_tune_cents[1] = x->bank_tune_cents[1];

    p->release_amt[0] = x->release_amt;
    p->release_amt[1] = x->release_amt2;
    p->stretch[0] = x->stretch;
    p->stretch[1] = x->stretch2;
    p->warp[0] = x->warp;
    p->warp[1] = x->warp2;
    p->brightness[0] = x->brightness;
    p->brightness[1] = x->brightness2;
    p->density_amt[0] = x->density_amt;
    p->density_amt[1] = x->density_amt2;
    p->density_mode[0] = (int)x->density_mode;
    p->density_mode[1] = (int)x->density_mode2;
    p->dispersion[0] = x->dispersion;
    p->dispersion[1] = x->dispersion2;
    p->odd_skew[0] = x->odd_skew;
    p->odd_skew[1] = x->odd_skew2;
    p->even_skew[0] = x->even_skew;
    p->even_skew[1] = x->even_skew2;
    p->collision_amt[0] = x->collision_amt;
    p->collision_amt[1] = x->collision_amt2;
    p->micro_detune[0] = x->micro_detune;
    p->micro_detune[1] = x->micro_detune2;

    p->excite_pos[0] = x->excite_pos;
    p->pickup_pos[0] = x->pickup_pos;
    p->excite_pos[1] = x->excite_pos2;
    p->pickup_pos[1] = x->pickup_pos2;

    for (int b = 0; b < 2; ++b){
        for (int d = 0; d < JB_N_DAMPERS; ++d){
            p->bell_peak_hz[b][d] = x->bell_peak_hz[b][d];
            p->bell_peak_zeta_param[b][d] = x->bell_peak_zeta_param[b][d];
            p->bell_npl[b][d] = x->bell_npl[b][d];
            p->bell_npr[b][d] = x->bell_npr[b][d];
            p->bell_npm[b][d] = x->bell_npm[b][d];
        }
    }

    p->space_size = x->space_size;
    p->space_decay = x->space_decay;
    p->space_diffusion = x->space_diffusion;
    p->space_damping = x->space_damping;
    p->space_wetdry = x->space_wetdry;

    p->exc_fader = x->exc_fader;
    p->exc_attack_ms = x->exc_attack_ms;
    p->exc_decay_ms = x->exc_decay_ms;
    p->exc_sustain = x->exc_sustain;
    p->exc_release_ms = x->exc_release_ms;
    p->exc_density = x->exc_density;
    p->exc_imp_shape = x->exc_imp_shape;
    p->exc_shape = x->exc_shape;

    p->lfo_index = x->lfo_index;
    for (int li = 0; li < JB_N_LFO; ++li){
        p->lfo_shape_v[li] = x->lfo_shape_v[li];
        p->lfo_rate_v[li]  = x->lfo_rate_v[li];
        p->lfo_phase_v[li] = x->lfo_phase_v[li];
        p->lfo_mode_v[li]  = x->lfo_mode_v[li];
        p->lfo_amt_v[li]   = x->lfo_amt_v[li];
        p->lfo_target[li]  = x->lfo_target[li];
    }

    p->velmap_amount = x->velmap_amount;
    for (int ti = 0; ti < JB_VELMAP_N_TARGETS; ++ti){
        p->velmap_on[ti] = x->velmap_on[ti];
    }
}

static void jb_preset_apply(t_juicy_bank_tilde *x, const jb_preset_t *p){
    if (!p || !p->used) return;

    x->bank_master[0] = p->bank_master[0];
    x->bank_master[1] = p->bank_master[1];
    x->bank_semitone[0] = p->bank_semitone[0];
    x->bank_semitone[1] = p->bank_semitone[1];
    x->bank_octave[0] = p->bank_octave[0];
    x->bank_octave[1] = p->bank_octave[1];
    x->bank_tune_cents[0] = p->bank_tune_cents[0];
    x->bank_tune_cents[1] = p->bank_tune_cents[1];

    x->release_amt  = p->release_amt[0];
    x->release_amt2 = p->release_amt[1];
    x->stretch  = p->stretch[0];
    x->stretch2 = p->stretch[1];
    x->warp  = p->warp[0];
    x->warp2 = p->warp[1];
    x->brightness  = p->brightness[0];
    x->brightness2 = p->brightness[1];
    x->density_amt  = p->density_amt[0];
    x->density_amt2 = p->density_amt[1];
    x->density_mode  = (jb_density_mode)p->density_mode[0];
    x->density_mode2 = (jb_density_mode)p->density_mode[1];
    x->dispersion  = p->dispersion[0];
    x->dispersion2 = p->dispersion[1];
    x->odd_skew  = p->odd_skew[0];
    x->odd_skew2 = p->odd_skew[1];
    x->even_skew  = p->even_skew[0];
    x->even_skew2 = p->even_skew[1];
    x->collision_amt  = p->collision_amt[0];
    x->collision_amt2 = p->collision_amt[1];
    x->micro_detune  = p->micro_detune[0];
    x->micro_detune2 = p->micro_detune[1];

    x->excite_pos  = p->excite_pos[0];
    x->pickup_pos  = p->pickup_pos[0];
    x->excite_pos2 = p->excite_pos[1];
    x->pickup_pos2 = p->pickup_pos[1];

    for (int b = 0; b < 2; ++b){
        for (int d = 0; d < JB_N_DAMPERS; ++d){
            x->bell_peak_hz[b][d] = p->bell_peak_hz[b][d];
            x->bell_peak_zeta_param[b][d] = p->bell_peak_zeta_param[b][d];
            // restore derived zeta if param is active
            if (x->bell_peak_zeta_param[b][d] >= 0.f){
                float u = x->bell_peak_zeta_param[b][d];
                u = jb_clamp(u, 0.f, 1.f);
                x->bell_peak_zeta[b][d] = jb_bell_map_norm_to_zeta(u);
            } else {
                x->bell_peak_zeta[b][d] = 0.f;
            }
            x->bell_npl[b][d] = p->bell_npl[b][d];
            x->bell_npr[b][d] = p->bell_npr[b][d];
            x->bell_npm[b][d] = p->bell_npm[b][d];
        }
    }

    x->space_size = p->space_size;
    x->space_decay = p->space_decay;
    x->space_diffusion = p->space_diffusion;
    x->space_damping = p->space_damping;
    x->space_wetdry = p->space_wetdry;

    x->exc_fader = p->exc_fader;
    x->exc_attack_ms = p->exc_attack_ms;
    x->exc_decay_ms = p->exc_decay_ms;
    x->exc_sustain = p->exc_sustain;
    x->exc_release_ms = p->exc_release_ms;
    x->exc_density = p->exc_density;
    x->exc_imp_shape = p->exc_imp_shape;
    x->exc_shape = p->exc_shape;

    for (int li = 0; li < JB_N_LFO; ++li){
        x->lfo_shape_v[li] = p->lfo_shape_v[li];
        x->lfo_rate_v[li]  = p->lfo_rate_v[li];
        x->lfo_phase_v[li] = p->lfo_phase_v[li];
        x->lfo_mode_v[li]  = p->lfo_mode_v[li];
        x->lfo_amt_v[li]   = p->lfo_amt_v[li];
        x->lfo_target[li]  = p->lfo_target[li] ? p->lfo_target[li] : gensym("none");
    }
    x->lfo_index = p->lfo_index;
    juicy_bank_tilde_lfo_index(x, x->lfo_index);

    x->velmap_amount = p->velmap_amount;
    for (int ti = 0; ti < JB_VELMAP_N_TARGETS; ++ti){
        x->velmap_on[ti] = p->velmap_on[ti];
    }
}

static void juicy_bank_tilde_preset_char(t_juicy_bank_tilde *x, t_floatarg f){
    if (!x || x->preset_mode != JB_PRESET_MODE_NAMING) return;
    int idx = (int)floorf(f + 0.5f);
    if (idx < 1) idx = 1;
    if (idx > 64) idx = 64;

    char c = jb_preset_char_from_index(idx);

    int cur = x->preset_cursor;
    if (cur < 0) cur = 0;
    if (cur >= JB_PRESET_NAME_MAX) cur = JB_PRESET_NAME_MAX - 1;

    x->preset_edit_name[cur] = c;
    x->preset_edit_name[JB_PRESET_NAME_MAX-1] = '\0';
    jb_preset_trim_name(x->preset_edit_name);

    if (x->out_preset){
        t_atom a[3];
        char chs[2]; chs[0] = c; chs[1] = 0;

        SETFLOAT(&a[0], (t_float)cur);
        SETSYMBOL(&a[1], gensym(chs));
        outlet_anything(x->out_preset, gensym("edit_char"), 2, a);

        SETSYMBOL(&a[0], gensym(x->preset_edit_name));
        outlet_anything(x->out_preset, gensym("preset_name"), 1, a);
    }
}

static void juicy_bank_tilde_preset_cmd(t_juicy_bank_tilde *x, t_symbol *s){
    if (!s) return;

    if (s == gensym("INIT")){
        if (x->preset_mode != JB_PRESET_MODE_NORMAL){
            // cancel/back-out
            x->preset_mode = JB_PRESET_MODE_NORMAL;
            x->preset_cursor = 0;
            x->preset_slot_sel = 0;
            memset(x->preset_edit_name, 0, sizeof(x->preset_edit_name));
            jb_preset_emit_ui(x);
            return;
        }
        // normal INIT: factory reset patch (existing init)
        juicy_bank_tilde_INIT(x);
        return;
    }

    if (s == gensym("SAVE")){
        if (x->preset_mode == JB_PRESET_MODE_NORMAL){
            x->preset_mode = JB_PRESET_MODE_NAMING;
            x->preset_cursor = 0;
            x->preset_slot_sel = 0;
            memset(x->preset_edit_name, ' ', JB_PRESET_NAME_MAX);
            x->preset_edit_name[JB_PRESET_NAME_MAX-1] = '\0';
            jb_preset_emit_ui(x);
            return;
        }
        if (x->preset_mode == JB_PRESET_MODE_NAMING){
            x->preset_mode = JB_PRESET_MODE_SLOT;
            x->preset_slot_sel = 0;
            jb_preset_emit_ui(x);
            return;
        }
        if (x->preset_mode == JB_PRESET_MODE_SLOT){
            int slot = x->preset_slot_sel;
            if (slot < 0) slot = 0;
            if (slot >= JB_PRESET_SLOTS) slot = JB_PRESET_SLOTS - 1;

            int overw = x->presets[slot].used;
            jb_preset_t *p = &x->presets[slot];
            jb_preset_snapshot(x, p);
            p->used = 1;
            strncpy(p->name, x->preset_edit_name, JB_PRESET_NAME_MAX);
            p->name[JB_PRESET_NAME_MAX] = '\0';
            jb_preset_trim_name(p->name);

            // exit
            x->preset_mode = JB_PRESET_MODE_NORMAL;

            if (x->out_preset){
                t_atom a[3];
                SETFLOAT(&a[0], (t_float)(slot + 1));
                SETSYMBOL(&a[1], gensym(p->name));
                outlet_anything(x->out_preset, gensym("preset_saved"), 2, a);
                if (overw){
                    outlet_anything(x->out_preset, gensym("overwrite"), 1, a);
                }
            }
            return;
        }
    }

    if (s == gensym("FORWARD") || s == gensym("BACKWARD")){
        int dir = (s == gensym("FORWARD")) ? +1 : -1;

        if (x->preset_mode == JB_PRESET_MODE_NAMING){
            x->preset_cursor += dir;
            if (x->preset_cursor < 0) x->preset_cursor = 0;
            if (x->preset_cursor >= JB_PRESET_NAME_MAX) x->preset_cursor = JB_PRESET_NAME_MAX - 1;
            if (x->out_preset){
                t_atom a;
                SETFLOAT(&a, (t_float)x->preset_cursor);
                outlet_anything(x->out_preset, gensym("edit_cursor"), 1, &a);
            }
            return;
        }

        if (x->preset_mode == JB_PRESET_MODE_SLOT){
            x->preset_slot_sel += dir;
            if (x->preset_slot_sel < 0) x->preset_slot_sel = JB_PRESET_SLOTS - 1;
            if (x->preset_slot_sel >= JB_PRESET_SLOTS) x->preset_slot_sel = 0;

            if (x->out_preset){
                t_atom a[2];
                SETFLOAT(&a[0], (t_float)(x->preset_slot_sel + 1));
                outlet_anything(x->out_preset, gensym("preset_slot"), 1, a);
                if (x->presets[x->preset_slot_sel].used){
                    outlet_anything(x->out_preset, gensym("overwrite"), 1, a);
                }
            }
            return;
        }

        // NORMAL: navigate through used presets and load them
        int cur = x->preset_slot_sel;
        if (cur < 0 || cur >= JB_PRESET_SLOTS) cur = 0;
        int nxt = jb_preset_find_next_used(x, cur, dir);
        if (nxt >= 0){
            x->preset_slot_sel = nxt;
            jb_preset_apply(x, &x->presets[nxt]);
            if (x->out_preset){
                t_atom a[2];
                SETFLOAT(&a[0], (t_float)(nxt + 1));
                SETSYMBOL(&a[1], gensym(x->presets[nxt].name));
                outlet_anything(x->out_preset, gensym("preset_loaded"), 2, a);
            }
        }
        return;
    }
}

// ---------- INIT (factory re-init) ----------
static void juicy_bank_tilde_INIT(t_juicy_bank_tilde *x){
    // Apply 32-mode saw defaults (1/n amplitude) to the *selected* bank, then reset states
    int b = (x->edit_bank != 0) ? 1 : 0;
    jb_apply_default_saw_bank(x, b);
    juicy_bank_tilde_restart(x);
    post("juicy_bank~: INIT complete (selected bank=%d, 32 modes, flat per-mode gains, brightness=0 -> saw slope, decay=1s).", b+1);
}
static void juicy_bank_tilde_init_alias(t_juicy_bank_tilde *x){ juicy_bank_tilde_INIT(x); }

// ---------- setup ----------
// === Partials / Index navigation helpers ===
static void juicy_bank_tilde_partials(t_juicy_bank_tilde *x, t_floatarg f){
    int K = (int)floorf(f + 0.5f);
    int *n_modes_p   = x->edit_bank ? &x->n_modes2      : &x->n_modes;
    int *active_p    = x->edit_bank ? &x->active_modes2 : &x->active_modes;
    int *edit_idx_p  = x->edit_bank ? &x->edit_idx2     : &x->edit_idx;

    if (K < 0) K = 0;
    if (K > *n_modes_p) K = *n_modes_p;
    *active_p = K;

    if (*active_p == 0) {
        *edit_idx_p = 0;
    } else if (*edit_idx_p >= *active_p) {
        *edit_idx_p = *active_p - 1;
    }
}

// --- Bank selection + per-bank master & semitone (STEP 1 scaffolding) ---
// bank: 1 = modal bank 1, 2 = modal bank 2 (edit focus only; DSP for bank2 comes in STEP 2)
static void juicy_bank_tilde_bank(t_juicy_bank_tilde *x, t_floatarg f){
    int b = (int)floorf(f + 0.5f);
    if (b < 1) b = 1;
    if (b > 2) b = 2;
    x->edit_bank = b - 1;
}



// master: per-bank output gain (0..1), written to selected bank
static void juicy_bank_tilde_master(t_juicy_bank_tilde *x, t_floatarg f){
    x->bank_master[x->edit_bank] = jb_clamp(f, 0.f, 1.f);
}

// octave: per-bank octave transpose (-2..+2), snapped to integer, written to selected bank
static void juicy_bank_tilde_octave(t_juicy_bank_tilde *x, t_floatarg f){
    int o = (int)floorf(f + 0.5f);
    if (o < -2) o = -2;
    if (o >  2) o =  2;
    x->bank_octave[x->edit_bank] = o;
}

// semitone: per-bank transpose (-12..+12), written to selected bank
static void juicy_bank_tilde_semitone(t_juicy_bank_tilde *x, t_floatarg f){
    int s = (int)floorf(f + 0.5f);
    if (s < -12) s = -12;
    if (s >  12) s =  12;
    x->bank_semitone[x->edit_bank] = s;
}

// tune: per-bank cents detune (-100..+100), written to selected bank
static void juicy_bank_tilde_tune(t_juicy_bank_tilde *x, t_floatarg f){
    x->bank_tune_cents[x->edit_bank] = jb_clamp(f, -100.f, 100.f);
}

static void juicy_bank_tilde_index_forward(t_juicy_bank_tilde *x){
    int *active_p   = x->edit_bank ? &x->active_modes2 : &x->active_modes;
    int *edit_idx_p = x->edit_bank ? &x->edit_idx2     : &x->edit_idx;
    int K = (*active_p > 0) ? *active_p : 1;
    *edit_idx_p = (*edit_idx_p + 1) % K;
    if (x->out_index) outlet_float(x->out_index, (t_float)(*edit_idx_p + 1));
}

static void juicy_bank_tilde_index_backward(t_juicy_bank_tilde *x){
    int *active_p   = x->edit_bank ? &x->active_modes2 : &x->active_modes;
    int *edit_idx_p = x->edit_bank ? &x->edit_idx2     : &x->edit_idx;
    int K = (*active_p > 0) ? *active_p : 1;
    *edit_idx_p = (*edit_idx_p - 1 + K) % K;
    if (x->out_index) outlet_float(x->out_index, (t_float)(*edit_idx_p + 1));
}

// ---------- CHECKPOINT: revert checkpoint for base gains/decays (does NOT bake damping) ----------


// -------------------------------------------------------------------------
// Modulation matrix helpers: parse selectors like "velocity_to_damping"
// -------------------------------------------------------------------------
static int jb_modmatrix_parse_selector(const char *name, int *src_out, int *tgt_out){
    if (!name) return 0;
    const char *sep = strstr(name, "_to_");
    if (!sep) return 0;

    size_t src_len = (size_t)(sep - name);
    size_t tgt_len = strlen(sep + 4);
    if (src_len == 0 || tgt_len == 0) return 0;

    char src[32];
    char tgt[32];
    if (src_len >= sizeof(src)) src_len = sizeof(src) - 1;
    if (tgt_len >= sizeof(tgt)) tgt_len = sizeof(tgt) - 1;

    memcpy(src, name, src_len);
    src[src_len] = '\0';
    memcpy(tgt, sep + 4, tgt_len);
    tgt[tgt_len] = '\0';

    int src_idx = -1;
    int tgt_idx = -1;

    // --- sources ---
    if (!strcmp(src, "velocity"))      src_idx = 0;
    else if (!strcmp(src, "pitch"))    src_idx = 1;
    else if (!strcmp(src, "lfo1"))     src_idx = 2;
    else if (!strcmp(src, "lfo2"))     src_idx = 3;
    else return 0;

    // --- targets ---
    if (!strcmp(tgt, "damping") || !strcmp(tgt, "damper"))              tgt_idx = 0;
    else if (!strcmp(tgt, "broadness") || !strcmp(tgt, "global_decay") || !strcmp(tgt, "globaldecay")) tgt_idx = 1;
    else if (!strcmp(tgt, "location") || !strcmp(tgt, "slope"))         tgt_idx = 2;
    else if (!strcmp(tgt, "brightness"))                                tgt_idx = 3;
    else if (!strcmp(tgt, "density"))                                   tgt_idx = 5;
    else if (!strcmp(tgt, "stretch"))                                   tgt_idx = 6;
    else if (!strcmp(tgt, "warp"))                                      tgt_idx = 7;
    else if (!strcmp(tgt, "odd_skew"))                                  tgt_idx = 8;
    else if (!strcmp(tgt, "even_skew"))                                 tgt_idx = 9;
    else if (!strcmp(tgt, "master"))                                    tgt_idx = 11;
    else if (!strcmp(tgt, "pitch"))                                     tgt_idx = 12;
    else if (!strcmp(tgt, "partials"))                                  tgt_idx = 14;
    else return 0;

    if (src_out) *src_out = src_idx;
    if (tgt_out) *tgt_out = tgt_idx;
    return 1;
}

// accept anything-style messages on the matrix inlet / left inlet
// e.g. "velocity_to_damping 0.5" or "lfo1_to_brightness -0.25"
static void juicy_bank_tilde_anything(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    if (!s) return;
    int src_idx = -1, tgt_idx = -1;
    if (!jb_modmatrix_parse_selector(s->s_name, &src_idx, &tgt_idx)) return;
    if (argc < 1) return;

    t_float amt = atom_getfloat(argv);
    if (amt < -1.f) amt = -1.f;
    else if (amt > 1.f) amt = 1.f;

    if (src_idx >= 0 && src_idx < JB_N_MODSRC &&
        tgt_idx >= 0 && tgt_idx < JB_N_MODTGT){
        x->mod_matrix[src_idx][tgt_idx] = amt;
    }
}


void juicy_bank_tilde_setup(void){

    // Cache symbols once (avoid gensym() in the audio thread)
    if(!jb_sym_master_1){
        jb_sym_master_1     = gensym("master_1");
        jb_sym_master_2     = gensym("master_2");
        jb_sym_lfo2_amount  = gensym("lfo2_amount");
        jb_sym_noise_timbre = gensym("noise_timbre");
        jb_sym_imp_shape    = gensym("imp_shape");
    }

    // Target-inlet proxy class (accepts bare selectors like 'brightness_1')
    if (!jb_tgtproxy_class){
        jb_tgtproxy_class = class_new(gensym("_jb_tgtproxy"),
                                      0, 0,
                                      sizeof(jb_tgtproxy),
                                      CLASS_PD, 0);
        class_addfloat(jb_tgtproxy_class, (t_method)jb_tgtproxy_float);
        class_addsymbol(jb_tgtproxy_class, (t_method)jb_tgtproxy_symbol);
        class_addlist(jb_tgtproxy_class, (t_method)jb_tgtproxy_list);
        class_addanything(jb_tgtproxy_class, (t_method)jb_tgtproxy_anything);
    }


    if (!jb_presetproxy_class){
        jb_presetproxy_class = class_new(gensym("_jb_presetproxy"),
                                         0, 0,
                                         sizeof(jb_presetproxy),
                                         CLASS_PD, 0);
        class_addsymbol(jb_presetproxy_class, (t_method)jb_presetproxy_symbol);
        class_addanything(jb_presetproxy_class, (t_method)jb_presetproxy_anything);
    }

    juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
                           (t_newmethod)juicy_bank_tilde_new,
                           (t_method)juicy_bank_tilde_free,
                           sizeof(t_juicy_bank_tilde), CLASS_DEFAULT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);

    // accept modulation-matrix configuration messages in two formats:
    // 1) Direct: "lfo1_to_pitch 0.5" (left inlet, via 'anything')
    // 2) Tagged: "matrix lfo1_to_pitch 0.5" (matrix inlet, via 'matrix' method)
    class_addanything(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anything);

    // BEHAVIOR
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_release, gensym("release"), A_DEFFLOAT, 0);

    // BODY (Type-4 Caughey bell damping)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_freq, gensym("bell_freq"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_zeta, gensym("bell_zeta"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_npl,  gensym("bell_npl"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_npr,  gensym("bell_npr"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_npm,  gensym("bell_npm"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damper_sel, gensym("damper_sel"), A_DEFFLOAT, 0);
    // Friendly aliases
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_freq, gensym("peak_freq"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_zeta, gensym("peak_zeta"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_npl,  gensym("left_pow"),   A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_npr,  gensym("right_pow"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bell_npm,  gensym("model_param"),A_DEFFLOAT, 0);

class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density, gensym("density"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_quantize, gensym("quantize"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0); // legacy alias
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_collision, gensym("collision"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_odd_skew,  gensym("odd_skew"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_even_skew, gensym("even_skew"), A_DEFFLOAT, 0);

    
    
    
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_stretch, gensym("stretch"), A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_warp, gensym("warp"), A_FLOAT, 0);
// Spatial coupling methods (excite/pickup + geometry)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position,     gensym("position"),      A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pickup,       gensym("pickup"),        A_DEFFLOAT, 0);
        class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_odd_even,     gensym("odd_even"),       A_DEFFLOAT, 0);
// legacy alias (sets both X and Y)
// LFO + ADSR methods
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_shape, gensym("lfo_shape"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_rate,  gensym("lfo_rate"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_phase, gensym("lfo_phase"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_mode,  gensym("lfo_mode"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_index, gensym("lfo_index"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_amount, gensym("lfo_amount"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo1_target, gensym("lfo1_target"), A_SYMBOL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo2_target, gensym("lfo2_target"), A_SYMBOL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_velmap_amount, gensym("velmap_amount"), A_DEFFLOAT, 0);

class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_preset_cmd,  gensym("preset_cmd"),  A_SYMBOL, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_preset_char, gensym("preset_char"), A_DEFFLOAT, 0);

// INDIVIDUAL (per-mode)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index, gensym("index"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_ratio_i, gensym("ratio"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_gain_i, gensym("gain"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay_i, gensym("decay"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay_i, gensym("decya"), A_DEFFLOAT, 0); // alias
// Lists & misc
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_modes, gensym("modes"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_active, gensym("active"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_freq, gensym("freq"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decays, gensym("decays"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_amps, gensym("amps"), A_GIMME, 0);

    // dispersion & seeds
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_seed, gensym("seed"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion_reroll, gensym("dispersion_reroll"), 0);

    // realism & misc
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_micro_detune, gensym("micro_detune"), A_DEFFLOAT, 0);

    // notes/poly (non-voice-specific)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note, gensym("note"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_midi, gensym("note_midi"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_voices, gensym("voices"), A_DEFFLOAT, 0);

    // voice-addressed (for [poly])
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_poly, gensym("note_poly"), A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_poly_midi, gensym("note_poly_midi"), A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);

    // base & reset
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_basef0, gensym("basef0"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_base_alias, gensym("base"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_restart, gensym("restart"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_preset_recall, gensym("preset"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_preset_recall, gensym("recall"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_INIT, gensym("INIT"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_init_alias, gensym("init"), 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_partials, gensym("partials"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_master,   gensym("master"),   A_DEFFLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_octave,   gensym("octave"),   A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_semitone, gensym("semitone"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_tune,     gensym("tune"),     A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bank,     gensym("bank"),     A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_space_size,      gensym("space_size"),      A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_space_decay,     gensym("space_decay"),     A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_space_diffusion, gensym("space_diffusion"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_space_damping,   gensym("space_damping"),   A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_space_wetdry,    gensym("space_wetdry"),    A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index_forward, gensym("forward"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index_backward, gensym("backward"), 0);

    
    // offline render (testing/regression)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_render, gensym("render"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_renderwrite, gensym("renderwrite"), A_SYMBOL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_renderclear, gensym("renderclear"), 0);

}
