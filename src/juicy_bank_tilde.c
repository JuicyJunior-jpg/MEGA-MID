// juicy_bank~ — modal resonator bank (V5.0)
// 5-voice poly + 2 tail voices, true stereo banks, Behavior + Body + Individual inlets.
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
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>


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
#define JB_MAX_VOICES    7
#define JB_ATTACK_VOICES 5
#define JB_TAIL_VOICES   (JB_MAX_VOICES - JB_ATTACK_VOICES)
#define JB_N_MODSRC 4
#define JB_N_MODTGT    15
#define JB_N_LFO       2
#define JB_N_DAMPERS  3
#define JB_PITCH_MOD_SEMITONES  2.0f

// ---------- SPACE (global stereo room) ----------
#define JB_SPACE_NCOMB     8
#define JB_SPACE_NCOMB_CH  4
#define JB_SPACE_MAX_DELAY 1700
#define JB_SPACE_NAP       4
#define JB_SPACE_AP_MAX    700


// ---------- utils ----------
static inline float jb_clamp(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }
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

// Feedback comb filter with 1-pole damping inside the feedback path.
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

    // branch matching state
    // NOTE: old exciter noise diffusion (all-pass cascade) removed.
}

// STEP 2 helpers (runtime reset)
static inline void jb_exc_voice_reset_runtime(jb_exc_voice_t *e){
    // Keep RNG seeds/states (stereo decorrelation persists), but clear all time-varying state.
    e->vel_cur = 0.f;
    e->vel_on  = 0.f;
    e->pitch   = 0.f;

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
    // Internal exciter runtime state (per voice, stereo)
    jb_exc_voice_t exc;
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


    // Feedback loop state (per voice, per bank, 1-sample delayed)
    // Each bank ONLY feeds back its own output.
    float fb_prevL[2], fb_prevR[2];         // last bank output (pre-space), L/R
    float fb_hp_x1L[2], fb_hp_y1L[2];       // 20Hz DC-blocker state (L)
    float fb_hp_x1R[2], fb_hp_y1R[2];       // 20Hz DC-blocker state (R)
    float fb_agc_gain[2];                   // AGC gain applied to feedback before reinjection
    float fb_agc_env[2];                    // AGC level follower (mono abs)

    // runtime per-mode — BANK 1/2
    jb_mode_rt_t m[JB_MAX_MODES];
    jb_mode_rt_t m2[JB_MAX_MODES];
} jb_voice_t;

// ---------- the object ----------
static t_class *juicy_bank_tilde_class;
static t_class *jb_tgtproxy_class;

// Proxy to accept ANY message on target-selection inlets (so message boxes like 'damper_1' work)
typedef struct _juicy_bank_tilde t_juicy_bank_tilde; // forward
typedef struct _jb_tgtproxy{
    t_pd p_pd;
    t_juicy_bank_tilde *owner;
    int lane; // 0=LFO1, 1=LFO2, 2=ADSR, 3=MIDI
} jb_tgtproxy;

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
    int   total_voices; // attack+tail voices actually processed
    jb_voice_t v[JB_MAX_VOICES];

    // current edit index for Individual setters
    int edit_idx;

    int edit_idx2;

    // RNG
    jb_rng_t rng;

    // DC HP
    float hp_a, hpL_x1, hpL_y1, hpR_x1, hpR_y1;
    float fb_hp_a; // 20Hz DC blocker inside Pressure feedback loop

    // SPACE parameters (0..1)
    float space_size;
    float space_decay;
    float space_diffusion;
    float space_damping;

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
    // Spatial excitation & pickup positions
    // - Excitation is 2D (x,y) using a simple membrane/plate-style shape product:
    //      ge(rank) = sin(nx*pi*x) * sin(ny*pi*y)    with (nx,ny) assigned per rank (diagonal mapping)
    // - Pickup is also 2D at a single mic point (x,y): gp(rank) = sin(nx*pi*x) * sin(ny*pi*y)
    float excite_pos;       // 0..1 (strike X position)
    float excite_pos_y;     // 0..1 (strike Y position)
    float pickup_x;      // 0..1 (mic X position)
    float pickup_y;      // 0..1 (mic Y position)

    // Bank 2 (independent spatial positions)
    float excite_pos2;      // 0..1 (strike X position, bank2)
    float excite_pos_y2;    // 0..1 (strike Y position, bank2)
    float pickup_x2;     // 0..1 (mic X position, bank2)
    float pickup_y2;     // 0..1 (mic Y position, bank2)
        // Odd vs Even emphasis (-1..+1). Applied as an index-based gain mask.
    float odd_even_bias;  // bank1
    float odd_even_bias2; // bank2
// inlet pointers for spatial position controls
    t_inlet *in_position;
    t_inlet *in_positionY;
    t_inlet *in_pickupX;
    t_inlet *in_pickupY;

        t_inlet *in_odd_even; // odd vs even emphasis
// --- LFO globals (for modulation matrix UI) ---
    // lfo_shape / lfo_rate / lfo_phase always reflect the *currently selected* LFO,
    // as chosen by lfo_index (1 or 2). Per-LFO values live in the arrays below.
    float lfo_shape;   // 1..4 (1=saw,2=square,3=sine,4=SH)
    float lfo_rate;    // 1..20 Hz
    float lfo_phase;   // 0..1
    float lfo_index;   // 1 or 2 (selects which LFO)

    // per-LFO parameter storage
    float lfo_shape_v[JB_N_LFO];
    float lfo_rate_v[JB_N_LFO];
    float lfo_phase_v[JB_N_LFO];

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
    t_symbol *lfo_target[JB_N_LFO];   // LFO1/LFO2 targets
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
    // per-block computed (shared)
    float exc_noise_color_gL, exc_noise_color_gH, exc_noise_color_comp;

    // --- FEEDBACK AGC (per-bank, per-voice) ---
    // Attack/Release are 0..1 UI parameters mapped to time constants (seconds).
    float fb_agc_attack;
    float fb_agc_release;

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
    t_inlet *in_exc_density;
    t_inlet *in_exc_imp_shape;
    t_inlet *in_exc_shape;

    // AGC inlets (placed after timbre/color, before LFO index)
    t_inlet *in_fb_agc_attack;
    t_inlet *in_fb_agc_release;

    // --- MOD SECTION inlets (targets/amounts; actual wiring added next step) ---
    t_inlet *in_lfo_index;
    t_inlet *in_lfo_shape;
    t_inlet *in_lfo_rate;
    t_inlet *in_lfo_phase;
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
        return jb_clamp(x->lfo_val[0], -1.f, 1.f);

    case 3: // lfo2 (-1..+1)
        return jb_clamp(x->lfo_val[1], -1.f, 1.f);

    default:
        return 0.f;
    }
}


// ---------- INTERNAL EXCITER (Fusion STEP 2) — block update + per-sample render ----------

// Update all per-voice exciter parameters that depend on the shared inlets.
// Called once per DSP block.
static void jb_exc_update_block(t_juicy_bank_tilde *x){
    float sr = (x->sr > 0.f) ? x->sr : 48000.f;

    // Impulse Shape -> HP/LP cutoffs (impulse only)
    float s = jb_clamp(x->exc_imp_shape, 0.f, 1.f);
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

    // Noise Color (slope EQ) -> per-block gains (shared)
    float color = jb_clamp(x->exc_shape, 0.f, 1.f); // 0=red .. 0.5=white .. 1=violet
    float slope_db_per_oct = -6.f + 12.f * color;
    float slope_db = slope_db_per_oct * JB_EXC_COLOR_OCT_SPAN;
    float gH = powf(10.f,  slope_db / 20.f);
    float gL = powf(10.f, -slope_db / 20.f);
    float comp = 1.f / sqrtf(0.5f * (gL*gL + gH*gH) + 1e-12f);
    x->exc_noise_color_gL = gL;
    x->exc_noise_color_gH = gH;
    x->exc_noise_color_comp = comp;
    // NOTE: old exciter noise diffusion (all-pass cascade) removed.

    // ADSR curves + times (applied to all voices)
    float aC = jb_clamp(x->exc_attack_curve,  -1.f, 1.f);
    float dC = jb_clamp(x->exc_decay_curve,   -1.f, 1.f);
    float rC = jb_clamp(x->exc_release_curve, -1.f, 1.f);

    float a_ms = x->exc_attack_ms;
    float d_ms = x->exc_decay_ms;
    float sus  = x->exc_sustain;
    float r_ms = x->exc_release_ms;
    for(int i=0; i<x->total_voices; ++i){
        jb_exc_voice_t *e = &x->v[i].exc;

        // Noise filters:
        //  - lpL/lpR: pivot low-pass for slope EQ (used to split low/high bands)
        //  - hpL/hpR: gentle DC high-pass (post diffusion)
        float pivot_hz = JB_EXC_SLOPE_PIVOT_HZ;
        if (pivot_hz < 50.f) pivot_hz = 50.f;
        if (pivot_hz > 0.45f * sr) pivot_hz = 0.45f * sr;
        jb_exc_lp1_set(&e->lpL, sr, pivot_hz);
        jb_exc_lp1_set(&e->lpR, sr, pivot_hz);
        jb_exc_hp1_set(&e->hpL, sr, 5.f);
        jb_exc_hp1_set(&e->hpR, sr, 5.f);

        // Impulse filters: shaped by the (new) impulse-only Shape inlet
        jb_exc_hp1_set(&e->hpImpL, sr, hp_hz);
        jb_exc_hp1_set(&e->hpImpR, sr, hp_hz);
        jb_exc_lp1_set(&e->lpImpL, sr, lp_hz);
        jb_exc_lp1_set(&e->lpImpR, sr, lp_hz);

        // env
        e->env.curveA = aC;
        e->env.curveD = dC;
        e->env.curveR = rC;
        jb_exc_adsr_set_times(&e->env, sr, a_ms, d_ms, sus, r_ms);
    }
}


// Update per-sample increments for the dedicated modulation ADSR.
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

    float colL = x->exc_noise_color_comp * (x->exc_noise_color_gL * lpL + x->exc_noise_color_gH * hpL);
    float colR = x->exc_noise_color_comp * (x->exc_noise_color_gL * lpR + x->exc_noise_color_gH * hpR);
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
// Brightness as spectral centroid tilt.
//
// Spectral centroid (center of mass):
//   C = sum(f_i * a_i) / sum(a_i)
//
// We implement the BRIGHTNESS parameter b in [-1, +1] as a frequency-dependent
// gain mask applied to the per-mode amplitudes:
//
//   a'_i = a_i * (f_i / C_ref)^(k)
//   k    = b * JB_BRIGHT_INTENSITY
//
// where C_ref is computed from the un-brightened spectrum (b = 0) for the
// currently active modes. This matches the intuition:
//   b = +1  -> boost highs (glassy / metallic)
//   b =  0  -> natural material spectrum
//   b = -1  -> damp highs (dark / muted)
//
// Loudness is kept stable later by normalizing the summed per-mode gains.
#ifndef JB_BRIGHT_INTENSITY
#define JB_BRIGHT_INTENSITY 2.0f
#endif
static inline float jb_bright_weight(float f_hz, float centroid_hz, float brightness){
    float b = jb_clamp(brightness, -1.f, 1.f);
    if (b == 0.f) return 1.f;

    float c = (centroid_hz > 0.f && isfinite(centroid_hz)) ? centroid_hz : 1.f;
    float f = (f_hz > 0.f && isfinite(f_hz)) ? f_hz : c;

    float k = b * JB_BRIGHT_INTENSITY;
    float w = powf(f / c, k);

    if (!isfinite(w) || w < 0.f) w = 0.f;

    // Prevent extreme weights (overall loudness is normalized later).
    if (w > 32.f) w = 32.f;

    return w;
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

static inline float jb_bell_zeta_eval(const t_juicy_bank_tilde *x, int bank, float omega){
    // Parameters (already mapped/clamped by inlet handlers)
    // Stacked Type-4 bell damping: zeta_total(omega) = sum_k zeta_k(omega)
    float zsum = 0.f;

    // avoid peaks above Nyquist (if sr is known)
    float sr = x->sr;
    float nyq = (sr > 1.f) ? (0.5f * sr) : 0.f;

    for (int k = 0; k < JB_N_DAMPERS; ++k){
        float zeta_p = jb_bank_bell_peak_zeta(x, bank, k);
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
static inline float jb_bank_excite_pos_y(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->excite_pos_y2 : x->excite_pos_y;
}

// Map a 0-based rank to a 2D (nx, ny) pair using diagonal enumeration:
//   rank 0 -> (1,1)
//   rank 1 -> (2,1), rank 2 -> (1,2)
//   rank 3 -> (3,1), rank 4 -> (2,2), rank 5 -> (1,3), ...
static inline void jb_rank_to_nm(int rank, int *nx, int *ny){
    int d = 2;        // diagonal index (sum = d)
    int idx = rank;   // remaining index within diagonals
    while (1){
        int count = d - 1; // number of pairs on this diagonal
        if (idx < count){
            int n = (d - 1) - idx;
            int m = d - n;
            *nx = n;
            *ny = m;
            return;
        }
        idx -= count;
        d++;
        if (d > 1024){ *nx = 1; *ny = 1; return; } // safety
    }
}
static inline float jb_bank_pickup_posL(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->pickup_x2 : x->pickup_x;
}
static inline float jb_bank_pickup_posR(const t_juicy_bank_tilde *x, int bank){
    return bank ? x->pickup_y2 : x->pickup_y;
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

    // Brightness: user-controlled (+ optional LFO1 modulation)
    {
        const t_symbol *lfo1_tgt = x->lfo_target[0];
        const float lfo1 = x->lfo_val[0] * x->lfo_amt_v[0];
        float b = jb_clamp(jb_bank_brightness(x, bank), -1.f, 1.f);
        if (lfo1 != 0.f){
            if ((bank == 0 && lfo1_tgt == gensym("brightness_1")) || (bank != 0 && lfo1_tgt == gensym("brightness_2"))){
                b = jb_clamp(b + lfo1, -1.f, 1.f);
            }
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
        const t_symbol *lfo1_tgt = x->lfo_target[0];
        const float lfo1 = x->lfo_val[0] * x->lfo_amt_v[0];
        if (lfo1 != 0.f){
            if ((bank == 0 && lfo1_tgt == gensym("density_1")) || (bank != 0 && lfo1_tgt == gensym("density_2"))){
                density_amt_eff += lfo1;
                if (density_amt_eff < -1.f) density_amt_eff = -1.f; // keep existing lower clamp behavior
            }
        }
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

// NEW MOD LANES (LFO1): bank pitch modulation in Hz, ±1 octave max depth
{
    const t_symbol *lfo1_tgt = x->lfo_target[0];
    const float lfo1 = x->lfo_val[0] * jb_clamp(x->lfo_amt_v[0], -1.f, 1.f);
    if (lfo1 != 0.f){
        if ((bank == 0 && lfo1_tgt == gensym("pitch_1")) || (bank != 0 && lfo1_tgt == gensym("pitch_2"))){
            // lfo1 is in [-1..+1] → map to frequency ratio [0.5..2.0] (one octave down/up)
            f0_eff *= powf(2.f, lfo1);
        }
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
        float zeta = jb_bell_zeta_eval(x, bank, omega);
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
    // NEW MOD LANES (partial): LFO1 direct-to-target modulation values
    const t_symbol *lfo1_tgt = x->lfo_target[0];
    const float lfo1 = x->lfo_val[0] * x->lfo_amt_v[0];
    // Odd vs Even emphasis bias (-1..+1): index-based mode gain mask
    const float odd_even = jb_clamp(bank ? x->odd_even_bias2 : x->odd_even_bias, -1.f, 1.f);


    // LFO1 -> partials_* : smooth float gating across active modes (0..active_count_idx)
    int   lfo1_partials_k = -1;
    float lfo1_partials_frac = 0.f;
    int   lfo1_partials_enabled = 0;

    float ratio_rel_sorted[JB_MAX_MODES];
    int   idx_sorted[JB_MAX_MODES];
    float order_t[JB_MAX_MODES];
    int   active_count = 0;

    // index-rank of each active mode (0..active_count_idx-1), in ascending index order
    int   rank_of_id[JB_MAX_MODES];
    int   freq_rank_of_id[JB_MAX_MODES];
    int   active_count_idx = 0;

    for (int i = 0; i < n_modes; ++i){
        order_t[i] = 0.f;
        rank_of_id[i] = -1;
        freq_rank_of_id[i] = -1;
        if (!base[i].active) continue;
        rank_of_id[i] = active_count_idx++;

        float ratio = m[i].ratio_now + disp_offset[i];
        float ratio_rel = base[i].keytrack ? ratio : ((v->f0 > 0.f) ? (ratio / v->f0) : ratio);
        if (ratio_rel < 0.f) ratio_rel = 0.f;

        ratio_rel_sorted[active_count] = ratio_rel;
        idx_sorted[active_count]      = i;
        active_count++;
    }

    for (int a = 0; a < active_count - 1; ++a){
        int   min_j   = a;
        float min_val = ratio_rel_sorted[a];
        for (int b = a + 1; b < active_count; ++b){
            if (ratio_rel_sorted[b] < min_val){
                min_val = ratio_rel_sorted[b];
                min_j   = b;
            }
        }
        if (min_j != a){
            float tmpv              = ratio_rel_sorted[a];
            ratio_rel_sorted[a]     = ratio_rel_sorted[min_j];
            ratio_rel_sorted[min_j] = tmpv;

            int tmpi          = idx_sorted[a];
            idx_sorted[a]     = idx_sorted[min_j];
            idx_sorted[min_j] = tmpi;
        }
    }

    if (active_count > 1){
        float denom = (float)(active_count - 1);
        for (int rank = 0; rank < active_count; ++rank){
            int i = idx_sorted[rank];
            order_t[i] = (float)rank / denom;
        }
    } else if (active_count == 1){
        order_t[idx_sorted[0]] = 0.f;
    }

    // Map each active mode ID -> rank by ascending frequency/ratio (Elements-style indexing).
    for (int rank = 0; rank < active_count; ++rank){
        int id = idx_sorted[rank];
        freq_rank_of_id[id] = rank;
    }

    // Elements-style position/pickup weighting uses pure mode index (n) after ordering by frequency.
    const float posx = jb_clamp(jb_bank_excite_pos(x, bank), 0.f, 1.f);
    float posy = jb_clamp(jb_bank_excite_pos_y(x, bank), 0.f, 1.f);
    const float micX = jb_clamp(jb_bank_pickup_posL(x, bank), 0.f, 1.f);
    const float micY = jb_clamp(jb_bank_pickup_posR(x, bank), 0.f, 1.f);

    float energy_pos = 0.f;
    const float PI   = (float)M_PI;
    for (int rank = 0; rank < active_count; ++rank){
        int nx, ny;
        jb_rank_to_nm(rank, &nx, &ny);
        const float ge = sinf((float)nx * PI * posx) * sinf((float)ny * PI * posy);
        energy_pos += ge * ge;
    }
    const float pos_norm = 1.f / sqrtf(energy_pos + 1e-5f);

    // Matching RSS normalization for pickup: keep output level stable as the mic point moves.
    // (Signed gains preserved; only the overall energy is normalized.)
    float energy_mic = 0.f;
    for (int rank = 0; rank < active_count; ++rank){
        int nx, ny;
        jb_rank_to_nm(rank, &nx, &ny);
        const float gp0 = sinf((float)nx * PI * micX) * sinf((float)ny * PI * micY);
        energy_mic += gp0 * gp0;
    }
    const float mic_norm = 1.f / sqrtf(energy_mic + 1e-5f);


    float brightness_v = bank ? v->brightness_v2 : v->brightness_v;

    // Brightness uses a spectral-centroid tilt; we normalize summed per-mode gains so
    // brightness redistributes energy without changing overall level.
    float sum_gain = 0.f;
    float sum_ref  = 0.f;

    // LFO1 -> partials_* : smooth float gating across active modes
    if (lfo1 != 0.f){
        if ((bank == 0 && lfo1_tgt == gensym("partials_1")) || (bank != 0 && lfo1_tgt == gensym("partials_2"))){
            float pf = (float)active_modes + lfo1 * (float)((active_count_idx > 1) ? (active_count_idx - 1) : 1);
            if (pf < 0.f) pf = 0.f;
            if (pf > (float)active_count_idx) pf = (float)active_count_idx;
            lfo1_partials_k = (int)floorf(pf);
            lfo1_partials_frac = pf - (float)lfo1_partials_k;
            lfo1_partials_enabled = 1;
        }
    }


// Reference spectral centroid (brightness = 0) for the currently active spectrum.
// C_ref = sum(f_i * a_i) / sum(a_i), using base gains (material) and partial-count gating.
float centroid_ref = (v->f0 > 0.f) ? v->f0 : 1.f;
{
    float sumA  = 0.f;
    float sumFA = 0.f;

    for (int i = 0; i < n_modes; ++i){
        if (!base[i].active) continue;

        float ratio = m[i].ratio_now + disp_offset[i];
        float ratio_rel = base[i].keytrack ? ratio : ((v->f0 > 0.f) ? (ratio / v->f0) : ratio);
        if (ratio_rel < 0.f) ratio_rel = 0.f;

        // Convert to Hz for centroid.
        float f_hz = base[i].keytrack ? (ratio_rel * ((v->f0 > 0.f) ? v->f0 : 1.f)) : ratio;

        float a = base[i].base_gain;

        // Apply nyquist kill (if set) as part of the active spectrum.
        if (m[i].nyq_kill) a = 0.f;

        // Apply partial-count gating (active_modes + smooth fade), same as main loop.
        if (active_modes <= 0){
            a = 0.f;
        } else if (active_modes < n_modes){
            int K = active_modes;
            if (i >= K){
                float fade_width = 3.f;
                float uu = ((float)i - (float)K) / fade_width;
                if (uu >= 1.f){
                    a = 0.f;
                } else if (uu > 0.f){
                    float w_fade = 0.5f * (1.f + cosf((float)M_PI * uu));
                    a *= w_fade;
                }
            }
        }

        // LFO1 partials gating (smooth float: 0..32) is in index-rank space.
        if (lfo1_partials_enabled){
            const int r = rank_of_id[i];
            float w_p = 0.f;
            if (r < lfo1_partials_k) w_p = 1.f;
            else if (r == lfo1_partials_k) w_p = lfo1_partials_frac;
            else w_p = 0.f;
            a *= w_p;
        }

        float aw = fabsf(a);
        if (aw <= 0.f) continue;

        if (!(f_hz > 0.f) || !isfinite(f_hz)) continue;

        sumA  += aw;
        sumFA += aw * f_hz;
    }

    if (sumA > 1e-12f){
        centroid_ref = sumFA / sumA;
        if (!isfinite(centroid_ref) || centroid_ref <= 0.f) centroid_ref = (v->f0 > 0.f) ? v->f0 : 1.f;
    }
}    for(int i = 0; i < n_modes; ++i){
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

        float f_hz = base[i].keytrack ? (ratio_rel * ((v->f0 > 0.f) ? v->f0 : 1.f)) : ratio;

        float g = base[i].base_gain * jb_bright_weight(f_hz, centroid_ref, brightness_v);

        float g_ref = base[i].base_gain;
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
        // --- Spatial coupling (2D excitation, 2D pickup; signed) ---
        float gnL = gn, gnR = gn;
        float gn_refL = gn_ref, gn_refR = gn_ref;
        {
            const int n = freq_rank_of_id[i]; // 0..active_count-1 (ascending frequency)
            if (n >= 0){
                // Excitation is 2D: ge = sin(nx*pi*posx) * sin(ny*pi*posy)
                int nx, ny;
                jb_rank_to_nm(n, &nx, &ny);
                const float ge  = (sinf((float)nx * PI * posx) * sinf((float)ny * PI * posy)) * pos_norm;

                // Pickup (mic) reads displacement at a single 2D mic position (X,Y).
                // We keep one pickup point for both L/R outputs (no stereo mic spread here).
                const float gp = (sinf((float)nx * PI * micX) * sinf((float)ny * PI * micY)) * mic_norm;

                const float wL = ge * gp;
                const float wR = ge * gp;
gnL     *= wL;
                gnR     *= wR;
                gn_refL *= wL;
                gn_refR *= wR;
            }
        }

        // Odd vs Even mask by mode index (1-based): odd => (1-bias), even => (1+bias)
        const int mode_num = i + 1;
        float oe = (mode_num & 1) ? (1.f - odd_even) : (1.f + odd_even);
        oe = jb_clamp(oe, 0.f, 1.f);
        gnL     *= oe;
        gnR     *= oe;
        gn_refL *= oe;
        gn_refR *= oe;

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
    // Feedback loop state (per bank)
    for (int b = 0; b < 2; ++b){
        v->fb_prevL[b] = v->fb_prevR[b] = 0.f;
        v->fb_hp_x1L[b] = v->fb_hp_y1L[b] = 0.f;
        v->fb_hp_x1R[b] = v->fb_hp_y1R[b] = 0.f;
        v->fb_agc_gain[b] = 1.f;
        v->fb_agc_env[b]  = 0.f;
    }
    v->rel_env  = 1.f;
    v->rel_env2 = 1.f;
    v->energy = 0.f;

    // Internal exciter reset (note-on reset + voice-steal reset)
    jb_exc_voice_reset_runtime(&v->exc);

    // BANK 1
    for(int i=0;i<x->n_modes;i++){
        jb_mode_rt_t *md=&v->m[i];
        md->ratio_now = x->base[i].base_ratio;
        md->decay_ms_now = x->base[i].base_decay_ms;
        md->gain_nowL = x->base[i].base_gain; md->gain_nowR = x->base[i].base_gain;
        md->t60_s = md->decay_ms_now*0.001f;
        jb_svf_reset(&md->svfL);
        jb_svf_reset(&md->svfR);
        md->driveL=md->driveR=0.f;
        md->y_pre_lastL=md->y_pre_lastR=0.f;
        md->hit_gateL=md->hit_gateR=0; md->hit_coolL=md->hit_coolR=0;
        md->md_hit_offsetL = 0.f; md->md_hit_offsetR = 0.f;
        md->nyq_kill = 0;

        v->disp_offset[i]=0.f; v->disp_target[i]=0.f;
                (void)rng;
    }

    // BANK 2
    for(int i=0;i<x->n_modes2;i++){
        jb_mode_rt_t *md=&v->m2[i];
        md->ratio_now = x->base2[i].base_ratio;
        md->decay_ms_now = x->base2[i].base_decay_ms;
        md->gain_nowL = x->base2[i].base_gain; md->gain_nowR = x->base2[i].base_gain;
        md->t60_s = md->decay_ms_now*0.001f;
        jb_svf_reset(&md->svfL);
        jb_svf_reset(&md->svfR);
        md->driveL=md->driveR=0.f;
        md->y_pre_lastL=md->y_pre_lastR=0.f;
        md->hit_gateL=md->hit_gateR=0; md->hit_coolL=md->hit_coolR=0;
        md->md_hit_offsetL = 0.f; md->md_hit_offsetR = 0.f;
        md->nyq_kill = 0;

        v->disp_offset2[i]=0.f; v->disp_target2[i]=0.f;
            }
}

static int jb_find_voice_to_steal(t_juicy_bank_tilde *x){
    int best=-1; float bestE=1e9f;
    for(int i=0;i<x->max_voices;i++){
        if (x->v[i].state==V_IDLE) return i;
        float e = x->v[i].energy;
        if (e<bestE){ bestE=e; best=i; }
    }
    return (best<0)?0:best;
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
        v->fb_prevL[b] = v->fb_prevR[b] = 0.f;
        v->fb_hp_x1L[b] = v->fb_hp_y1L[b] = 0.f;
        v->fb_hp_x1R[b] = v->fb_hp_y1R[b] = 0.f;
        v->fb_agc_gain[b] = 1.f;
        v->fb_agc_env[b]  = 0.f;
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
static int jb_find_idle_tail_voice(const t_juicy_bank_tilde *x){
    if (!x) return -1;
    int start = x->max_voices;
    int end   = x->total_voices;
    if (start < 0) start = 0;
    if (start > JB_MAX_VOICES) start = JB_MAX_VOICES;
    if (end < 0) end = 0;
    if (end > JB_MAX_VOICES) end = JB_MAX_VOICES;
    if (end <= start) return -1;

    for(int i=start; i<end; ++i){
        if (x->v[i].state == V_IDLE) return i;
    }
    return -1;
}
static int jb_find_quietest_tail_voice(const t_juicy_bank_tilde *x){
    if (!x) return -1;
    int start = x->max_voices;
    int end   = x->total_voices;
    if (start < 0) start = 0;
    if (start > JB_MAX_VOICES) start = JB_MAX_VOICES;
    if (end < 0) end = 0;
    if (end > JB_MAX_VOICES) end = JB_MAX_VOICES;
    if (end <= start) return -1;

    int best = start;
    float bestE = 1e9f;

    for(int i=start; i<end; ++i){
        // Prefer stealing already-low-energy tails; if an idle slot exists, use it immediately.
        if (x->v[i].state == V_IDLE) return i;
        float e = x->v[i].energy;
        if (e < bestE){ bestE = e; best = i; }
    }
    return best;
}

static void jb_note_on(t_juicy_bank_tilde *x, float f0, float vel){
    // Pick an ATTACK voice (0..max_voices-1). Tail voices are never selected for new attacks.
    int idx = jb_find_voice_to_steal(x);
    jb_voice_t *v = &x->v[idx];

    // If we're stealing a currently-sounding voice, migrate it into the tail pool so it can
    // ring out naturally (as if a key was released) while we reuse this attack slot immediately.
    if (v->state != V_IDLE){
        int tidx = jb_find_idle_tail_voice(x);
        if (tidx < 0) tidx = jb_find_quietest_tail_voice(x);
        jb_voice_t *t = &x->v[tidx];

        *t = *v; // POD copy: includes resonator states + exciter runtime + env state

        // Convert copied voice to a natural NOTE-OFF release tail.
        jb_exc_note_off(t);
        t->state = V_RELEASE;
        // keep rel_env/rel_env2 as-is (they're 1 while held/releasing), they will decay in DSP loop
    }

    // Start (or restart) the attack voice immediately
    v->state = V_HELD;
    v->f0  = (f0<=0.f)?1.f:f0;
    // Velocity is accepted as either 0..1 or MIDI 0..127, and always stored 0..1.
    v->vel = jb_exc_midi_to_vel01(vel);

    jb_voice_reset_states(x, v, &x->rng);
    jb_project_behavior_into_voice(x, v);
    jb_project_behavior_into_voice2(x, v);
    jb_exc_note_on(x, v, v->vel);
}



// ===== Explicit voice-addressed control (for Pd [poly]) =====
static inline void jb_migrate_to_tail_if_needed(t_juicy_bank_tilde *x, const jb_voice_t *src){
    if (!x || !src) return;
    if (src->state == V_IDLE) return;

    int tidx = jb_find_idle_tail_voice(x);
    if (tidx < 0) tidx = jb_find_quietest_tail_voice(x);
    if (tidx < 0 || tidx >= x->total_voices) return;

    jb_voice_t *t = &x->v[tidx];
    *t = *src; // deep copy (struct contains full per-mode state)

    // Convert copied voice to a natural NOTE-OFF release tail.
    jb_exc_note_off(t);
    t->state = V_RELEASE;

    // Ensure tails are audible even if the stolen voice was already in release.
    t->rel_env  = 1.f;
    t->rel_env2 = 1.f;
}

static void jb_note_on_voice(t_juicy_bank_tilde *x, int vix1, float f0, float vel){
    // This path is used by [note_poly]/[poly]-style voice addressing.
    // We still want tail behavior here: stealing a busy attack voice migrates it into the tail pool.
    if (vix1 < 1) vix1 = 1;
    if (vix1 > x->max_voices) vix1 = x->max_voices;
    int idx = vix1 - 1;

    if (f0 <= 0.f) f0 = 1.f;
    // Velocity is accepted as either 0..1 or MIDI 0..127.
    vel = jb_exc_midi_to_vel01(vel);

    jb_voice_t *v = &x->v[idx];

    // If this voice is currently sounding, migrate it into a tail slot so it can ring out naturally.
    jb_migrate_to_tail_if_needed(x, v);

    // Start the new note immediately in the requested attack slot.
    v->state = V_HELD;
    v->f0 = f0;
    v->vel = vel;

    jb_voice_reset_states(x, v, &x->rng);
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

    // clear outputs
    for(int i=0;i<n;i++){ outL[i]=0; outR[i]=0; }

    // update LFOs once per block (for modulation matrix sources)
    jb_update_lfos_block(x, n);

    // NEW MOD LANES: LFO1 output (scaled by its amount) + a few global mods that must happen pre-exciter-update
    const t_symbol *lfo1_tgt = x->lfo_target[0];
    const float lfo1_amt = jb_clamp(x->lfo_amt_v[0], -1.f, 1.f);
    const float lfo1_out = x->lfo_val[0] * lfo1_amt;

    // store effective LFO amounts for downstream use
    x->lfo_amt_eff[0] = lfo1_amt;
    x->lfo_amt_eff[1] = jb_clamp(x->lfo_amt_v[1], -1.f, 1.f);
    if (lfo1_out != 0.f && lfo1_tgt == gensym("lfo2_amount")) {
        x->lfo_amt_eff[1] = jb_clamp(x->lfo_amt_eff[1] + lfo1_out, -1.f, 1.f);
    }

    // LFO1 -> Exciter params (0..1, additive, clamped) must be applied before jb_exc_update_block()
    const float exc_shape_saved = x->exc_shape;         // Noise Color
    const float exc_imp_shape_saved = x->exc_imp_shape; // Impulse Shape
    if (lfo1_out != 0.f){
        if (lfo1_tgt == gensym("exc_shape")) {
            x->exc_shape = jb_clamp(exc_shape_saved + lfo1_out, 0.f, 1.f);
        } else if (lfo1_tgt == gensym("exc_imp_shape")) {
            x->exc_imp_shape = jb_clamp(exc_imp_shape_saved + lfo1_out, 0.f, 1.f);
        }
    }

    // Internal exciter: update shared params -> per-voice filters + ADSR times/curves
    jb_exc_update_block(x);
    x->exc_shape = exc_shape_saved;
    x->exc_imp_shape = exc_imp_shape_saved;

        // (Coupling removed) Both banks are always excited by the internal exciter and always summed to the output.
    // Internal exciter mix weights (computed once per block)
    float exc_f = jb_clamp(x->exc_fader, -1.f, 1.f);
    float exc_t = 0.5f * (exc_f + 1.f);
    float exc_w_imp   = sinf(0.5f * (float)M_PI * exc_t);
    float exc_w_noise = cosf(0.5f * (float)M_PI * exc_t);
    // Pressure (reuses former density inlet): feedback-loop AGC target level (0..0.98)
    const float pressure = jb_clamp(x->exc_density, 0.f, 1.f);
    const float fb_target = 0.98f * pressure;

    // Feedback AGC smoothing coefficients (per block)
    // Attack:  0..1 -> 1ms..200ms  (fast clamp-down)
    // Release: 0..1 -> 10ms..2500ms (slow recovery)
    const float fb_tauA = 0.001f + 0.199f * jb_clamp(x->fb_agc_attack,  0.f, 1.f);
    const float fb_tauR = 0.010f + 2.490f * jb_clamp(x->fb_agc_release, 0.f, 1.f);
    const float a_fb_attack  = expf(-1.0f / (x->sr * fb_tauA));
    const float a_fb_release = expf(-1.0f / (x->sr * fb_tauR));
    const float one_minus_a_fb_attack  = 1.f - a_fb_attack;
    const float one_minus_a_fb_release = 1.f - a_fb_release;

    // Level detector (abs follower) for AGC: fixed 10ms time constant
    const float a_fb_env = expf(-1.0f / (x->sr * 0.010f));
    const float one_minus_a_fb_env = 1.f - a_fb_env;
    // Energy meter (per voice) used for voice stealing and tail cleanup.
    // 50ms time constant -> responsive but stable.
    const float a_energy = expf(-1.0f / (x->sr * 0.050f));
    const float one_minus_a_energy = 1.f - a_energy;
    const float tail_energy_thresh = 1e-6f;

    // Per-block updates that don't change sample-phase
    for(int vix=0; vix<x->total_voices; ++vix){
        jb_voice_t *v = &x->v[vix];
        if (v->state==V_IDLE) continue;
        jb_project_behavior_into_voice(x, v); // keep behavior up-to-date
        jb_update_voice_coeffs(x, v);
        jb_update_voice_gains(x, v);
        // bank 2 runtime prep (render/mix happens in STEP 2B-2)
        jb_project_behavior_into_voice2(x, v);
        jb_update_voice_coeffs2(x, v);
        jb_update_voice_gains2(x, v);
    }

    // constants

    // Process per-voice, sample-major so feedback uses only a 2-sample delay (no block latency)
    for(int vix=0; vix<x->total_voices; ++vix){
        jb_voice_t *v = &x->v[vix];
        if (v->state==V_IDLE) continue;

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

        // LFO1 -> master_* (bank output volume): additive + clamp (0..1)
        if (lfo1_out != 0.f){
            if (lfo1_tgt == gensym("master_1")){
                bank_gain1 = jb_clamp(bank_gain1 + lfo1_out, 0.f, 1.f);
            }else if (lfo1_tgt == gensym("master_2")){
                bank_gain2 = jb_clamp(bank_gain2 + lfo1_out, 0.f, 1.f);
            }
        }

        // Pan is intentionally not used in this synth anymore.

        const jb_mode_base_t *base1 = x->base;
        const jb_mode_base_t *base2 = x->base2;

        for(int i=0;i<n;i++){
            // Per-bank voice outputs (pre-space)
            float b1OutL = 0.f, b1OutR = 0.f;
            float b2OutL = 0.f, b2OutR = 0.f;
            // ---------- FEEDBACK (per-bank, per-voice) ----------
            // Each bank ONLY feeds back its own output. Feedback is injected at the
            // same junction as the exciter input, but it does NOT pass through the
            // exciter ADSR/noise path.
            float fb1L = 0.f, fb1R = 0.f;
            float fb2L = 0.f, fb2R = 0.f;
            {
                // Bank 1 (index 0)
                float inL = jb_hp1_run_a(v->fb_prevL[0], x->fb_hp_a, &v->fb_hp_x1L[0], &v->fb_hp_y1L[0]);
                float inR = jb_hp1_run_a(v->fb_prevR[0], x->fb_hp_a, &v->fb_hp_x1R[0], &v->fb_hp_y1R[0]);
                inL = jb_softclip_thresh(inL, 0.75f);
                inR = jb_softclip_thresh(inR, 0.75f);
                float lvl = 0.5f * (fabsf(inL) + fabsf(inR));
                v->fb_agc_env[0] = jb_kill_denorm(a_fb_env * v->fb_agc_env[0] + one_minus_a_fb_env * lvl);
                float desired = (fb_target <= 1e-9f) ? 0.f : (fb_target / (v->fb_agc_env[0] + 1e-6f));
                if (desired > 0.98f) desired = 0.98f;
                if (desired < 0.f) desired = 0.f;
                float g = v->fb_agc_gain[0];
                if (desired < g) g = a_fb_attack * g + one_minus_a_fb_attack * desired;
                else             g = a_fb_release * g + one_minus_a_fb_release * desired;
                v->fb_agc_gain[0] = g;
                fb1L = inL * g;
                fb1R = inR * g;
                // Final safety saturator (ceiling ~= 0.99)
                if (fb1L > 0.99f) fb1L = 0.99f; else if (fb1L < -0.99f) fb1L = -0.99f;
                if (fb1R > 0.99f) fb1R = 0.99f; else if (fb1R < -0.99f) fb1R = -0.99f;
            }
            {
                // Bank 2 (index 1)
                float inL = jb_hp1_run_a(v->fb_prevL[1], x->fb_hp_a, &v->fb_hp_x1L[1], &v->fb_hp_y1L[1]);
                float inR = jb_hp1_run_a(v->fb_prevR[1], x->fb_hp_a, &v->fb_hp_x1R[1], &v->fb_hp_y1R[1]);
                inL = jb_softclip_thresh(inL, 0.75f);
                inR = jb_softclip_thresh(inR, 0.75f);
                float lvl = 0.5f * (fabsf(inL) + fabsf(inR));
                v->fb_agc_env[1] = jb_kill_denorm(a_fb_env * v->fb_agc_env[1] + one_minus_a_fb_env * lvl);
                float desired = (fb_target <= 1e-9f) ? 0.f : (fb_target / (v->fb_agc_env[1] + 1e-6f));
                if (desired > 0.98f) desired = 0.98f;
                if (desired < 0.f) desired = 0.f;
                float g = v->fb_agc_gain[1];
                if (desired < g) g = a_fb_attack * g + one_minus_a_fb_attack * desired;
                else             g = a_fb_release * g + one_minus_a_fb_release * desired;
                v->fb_agc_gain[1] = g;
                fb2L = inL * g;
                fb2R = inR * g;
                if (fb2L > 0.99f) fb2L = 0.99f; else if (fb2L < -0.99f) fb2L = -0.99f;
                if (fb2R > 0.99f) fb2R = 0.99f; else if (fb2R < -0.99f) fb2R = -0.99f;
            }

            // ---------- INTERNAL EXCITER (no feedback mixed into noise/impulse) ----------
            float ex0L = 0.f, ex0R = 0.f;
            jb_exc_process_sample(x, v,
                                 exc_w_imp, exc_w_noise,
                                 &ex0L, &ex0R);
            ex0L = jb_kill_denorm(ex0L);
            ex0R = jb_kill_denorm(ex0R);
            if (!jb_isfinitef(ex0L) || !jb_isfinitef(ex0R)) { ex0L = 0.f; ex0R = 0.f; }

            // BANK 2 input: exciter + bank2 feedback
            float exL = ex0L + fb2L;
            float exR = ex0R + fb2R;
            // -------- BANK 2 --------
            if (bank_gain2 > 0.f && v->rel_env2 > 0.f){
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
            }
            // BANK 1 input: exciter + bank1 feedback
            exL = ex0L + fb1L;
            exR = ex0R + fb1R;

// -------- BANK 1 --------
            if (bank_gain1 > 0.f && v->rel_env > 0.f){
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
            }
            // Final per-voice sum (pre-space)
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
            // Update feedback sources for next sample (per bank, per voice)
            v->fb_prevL[0] = jb_kill_denorm(b1OutL);
            v->fb_prevR[0] = jb_kill_denorm(b1OutR);
            v->fb_prevL[1] = jb_kill_denorm(b2OutL);
            v->fb_prevR[1] = jb_kill_denorm(b2OutR);

            outL[i] += vOutL;
            outR[i] += vOutR;

            // Energy meter (used for stealing + tail cleanup). Uses abs-sum with 50ms smoothing.
            {
                float e_in = fabsf(vOutL) + fabsf(vOutR);
                if (!jb_isfinitef(e_in)) e_in = 0.f;
                v->energy = jb_kill_denorm(a_energy * v->energy + one_minus_a_energy * e_in);
            }

            // Release handling:
            // - Attack voices (0..max_voices-1): use the classic release_amt fade.
            // - Tail voices (max_voices..total_voices-1): NO extra fade (natural decay); freed when silent.
            if (v->state == V_RELEASE){
                const int is_tail = (vix >= x->max_voices);

                // Apply release envelope to ALL voices (including tails), so the RELEASE control
                // behaves consistently even during fast re-triggers that migrate voices into the tail pool.
                float tau1 = 0.02f + 4.98f * jb_clamp(x->release_amt,  0.f, 1.f);
                float tau2 = 0.02f + 4.98f * jb_clamp(x->release_amt2, 0.f, 1.f);
                float a_rel1 = expf(-1.0f / (x->sr * tau1));
                float a_rel2 = expf(-1.0f / (x->sr * tau2));

                v->rel_env  *= a_rel1;
                v->rel_env2 *= a_rel2;
                if (v->rel_env  < 1e-5f) v->rel_env  = 0.f;
                if (v->rel_env2 < 1e-5f) v->rel_env2 = 0.f;

                // Fast path: if envelopes hit zero, free immediately.
                if (v->rel_env == 0.f && v->rel_env2 == 0.f){
                    v->state = V_IDLE;
                    if (is_tail){
                        v->energy = 0.f;
                        for (int b = 0; b < 2; ++b){
                            v->fb_prevL[b] = v->fb_prevR[b] = 0.f;
                            v->fb_hp_x1L[b] = v->fb_hp_y1L[b] = 0.f;
                            v->fb_hp_x1R[b] = v->fb_hp_y1R[b] = 0.f;
                            v->fb_agc_gain[b] = 1.f;
                            v->fb_agc_env[b] = 0.f;
                        }
                    }
                } else if (is_tail){
                    // Tail cleanup: once the exciter envelope is idle and output energy is gone, free the tail voice.
                    if (v->exc.env.stage == JB_EXC_ENV_IDLE && v->energy < tail_energy_thresh){
                        v->state = V_IDLE;
                        v->energy = 0.f;
                        for (int b = 0; b < 2; ++b){
                            v->fb_prevL[b] = v->fb_prevR[b] = 0.f;
                            v->fb_hp_x1L[b] = v->fb_hp_y1L[b] = 0.f;
                            v->fb_hp_x1R[b] = v->fb_hp_y1R[b] = 0.f;
                            v->fb_agc_gain[b] = 1.f;
                            v->fb_agc_env[b] = 0.f;
                        }
                    }
                }


            } else if (v->state == V_HELD) {
                v->rel_env  = 1.f;
                v->rel_env2 = 1.f;
            } else {
                v->rel_env  = 0.f;
                v->rel_env2 = 0.f;
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
    // Parameters are 0..1 floats mapped exactly as specified.
    {
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

        // Fixed wet/dry mix (no inlet in this revision)
        const float mix = 0.35f;
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
    int b = x->edit_bank ? 1 : 0;
    int d = x->edit_damper;
    if (d < 0) d = 0; else if (d >= JB_N_DAMPERS) d = JB_N_DAMPERS - 1;
    x->bell_peak_zeta[b][d] = v;
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
// --- Spatial coupling param setters (gain-level only) ---
static void juicy_bank_tilde_position_x(t_juicy_bank_tilde *x, t_floatarg f){
    // excitation X position on the 2D surface (0..1)
    float v = jb_clamp(f, 0.f, 1.f);
    if (x->edit_bank) x->excite_pos2 = v;
    else              x->excite_pos  = v;
}
static void juicy_bank_tilde_position_y(t_juicy_bank_tilde *x, t_floatarg f){
    // excitation Y position on the 2D surface (0..1)
    float v = jb_clamp(f, 0.f, 1.f);
    if (x->edit_bank) x->excite_pos_y2 = v;
    else              x->excite_pos_y  = v;
}
// Legacy alias: "position" == "position_x"
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){
    juicy_bank_tilde_position_x(x, f);
}


// ---------- SPACE setters (0..1) ----------
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


static void juicy_bank_tilde_pickupL(t_juicy_bank_tilde *x, t_floatarg f){
    // pickup/mic X position along the 1D object (0..1)
    float v = jb_clamp(f, 0.f, 1.f);
    if (x->edit_bank) x->pickup_x2 = v;
    else              x->pickup_x  = v;
}
static void juicy_bank_tilde_pickupR(t_juicy_bank_tilde *x, t_floatarg f){
    // pickup/mic Y position along the 1D object (0..1)
    float v = jb_clamp(f, 0.f, 1.f);
    if (x->edit_bank) x->pickup_y2 = v;
    else              x->pickup_y  = v;
}

static void juicy_bank_tilde_odd_even(t_juicy_bank_tilde *x, t_floatarg f){
    // Odd vs Even emphasis bias: -1..+1
    // -1 => silence even modes, 0 => neutral, +1 => silence odd modes
    float v = jb_clamp(f, -1.f, 1.f);
    if (x->edit_bank) x->odd_even_bias2 = v;
    else              x->odd_even_bias  = v;
}

static void juicy_bank_tilde_pickup(t_juicy_bank_tilde *x, t_floatarg f){
    // legacy: set BOTH pickup coordinates (X and Y) (0..1)
    float v = jb_clamp(f, 0.f, 1.f);
    if (x->edit_bank){ x->pickup_x2 = v; x->pickup_y2 = v; }
    else             { x->pickup_x  = v; x->pickup_y  = v; }
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

static inline int jb_lfo1_target_allowed(t_symbol *s){
    if (jb_target_is_none(s)) return 1;
    return (
        s == gensym("master_1") || s == gensym("master_2") ||
        s == gensym("pitch_1")  || s == gensym("pitch_2")  ||
        s == gensym("brightness_1") || s == gensym("brightness_2") ||
        s == gensym("density_1")    || s == gensym("density_2")    ||
        s == gensym("partials_1")   || s == gensym("partials_2")   ||
        s == gensym("exc_shape")    ||
        s == gensym("exc_imp_shape")||
        s == gensym("lfo2_rate")    ||
        s == gensym("lfo2_amount")
    );
}

static void juicy_bank_tilde_lfo1_target(t_juicy_bank_tilde *x, t_symbol *s){
    if (!s) return;
    if (!jb_lfo1_target_allowed(s)){
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
    if (jb_target_is_none(s)){
        x->lfo_target[1] = gensym("none");
        return;
    }
    if (jb_target_taken(x, s, 1)) return;
    x->lfo_target[1] = s;
}





// ---------- target inlet proxy (accepts bare selectors like 'brightness_1') ----------
static void jb_tgtproxy_set(jb_tgtproxy *p, t_symbol *tgt){
    if (!p || !p->owner || !tgt) return;
    switch(p->lane){
        case 0: juicy_bank_tilde_lfo1_target(p->owner, tgt); break;
        case 1: juicy_bank_tilde_lfo2_target(p->owner, tgt); break;
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
    (void)nf; x->max_voices = JB_ATTACK_VOICES;
    x->total_voices = JB_MAX_VOICES;
// fixed playable/tail split
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
            x->v[v].fb_prevL[b] = x->v[v].fb_prevR[b] = 0.f;
            x->v[v].fb_hp_x1L[b] = x->v[v].fb_hp_y1L[b] = 0.f;
            x->v[v].fb_hp_x1R[b] = x->v[v].fb_hp_y1R[b] = 0.f;
            x->v[v].fb_agc_gain[b] = 1.f;
            x->v[v].fb_agc_env[b] = 0.f;
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
            x->v[v].fb_prevL[b] = x->v[v].fb_prevR[b] = 0.f;
            x->v[v].fb_hp_x1L[b] = x->v[v].fb_hp_y1L[b] = 0.f;
            x->v[v].fb_hp_x1R[b] = x->v[v].fb_hp_y1R[b] = 0.f;
            x->v[v].fb_agc_gain[b] = 1.f;
            x->v[v].fb_agc_env[b] = 0.f;
        }
    }
}

// ---------- dsp setup/free ----------
static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;
    float fc=8.f;  float RC=1.f/(2.f*M_PI*fc);  float dt=1.f/x->sr; x->hp_a=RC/(RC+dt);
    float fc_fb=20.f; float RCfb=1.f/(2.f*M_PI*fc_fb); x->fb_hp_a=RCfb/(RCfb+dt);

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
    inlet_free(x->in_positionY);
inlet_free(x->in_pickupX);
    inlet_free(x->in_pickupY);
    inlet_free(x->in_odd_even);
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
    if (x->in_exc_density) inlet_free(x->in_exc_density);
    if (x->in_exc_imp_shape) inlet_free(x->in_exc_imp_shape);
    if (x->in_exc_shape) inlet_free(x->in_exc_shape);
    if (x->in_fb_agc_attack) inlet_free(x->in_fb_agc_attack);
    if (x->in_fb_agc_release) inlet_free(x->in_fb_agc_release);

    // MOD SECTION inlets
    if (x->in_lfo_index) inlet_free(x->in_lfo_index);
    if (x->in_lfo_shape) inlet_free(x->in_lfo_shape);
    if (x->in_lfo_rate) inlet_free(x->in_lfo_rate);
    if (x->in_lfo_phase) inlet_free(x->in_lfo_phase);
    if (x->in_lfo_amount) inlet_free(x->in_lfo_amount);
    if (x->in_lfo1_target) inlet_free(x->in_lfo1_target);
    if (x->in_lfo2_target) inlet_free(x->in_lfo2_target);

    // Target proxies
    if (x->tgtproxy_lfo1) { pd_free((t_pd *)x->tgtproxy_lfo1); x->tgtproxy_lfo1 = 0; }
    if (x->tgtproxy_lfo2) { pd_free((t_pd *)x->tgtproxy_lfo2); x->tgtproxy_lfo2 = 0; }


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
            x->bell_peak_zeta[0][d] = (d == 0) ? 0.00035f : 0.f; // damper 1 active, others off by default
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

    // body defaults
    x->brightness=0.f; x->density_amt=0.f; x->density_mode=DENSITY_PIVOT;
    x->dispersion=0.f; x->dispersion_last=-1.f;

    // Type-4 bell damping defaults (stackable 3x; damper 1 active, others off)
    for (int d = 0; d < JB_N_DAMPERS; ++d){
        x->bell_peak_hz[0][d]   = 3000.f;
        x->bell_peak_zeta[0][d] = (d == 0) ? 0.00035f : 0.f;
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
    x->excite_pos=0.33f; x->excite_pos_y=0.33f; x->pickup_x=0.33f; x->pickup_y=0.33f;
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
x->excite_pos_y2  = x->excite_pos_y;
    x->pickup_x2   = x->pickup_x;
    x->pickup_y2   = x->pickup_y;

    // LFO + ADSR defaults
    x->lfo_shape = 1.f;   // default: shape 1 (for currently selected LFO)
    x->lfo_rate  = 1.f;   // 1 Hz
    x->lfo_phase = 0.f;   // start at phase 0
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

    x->exc_density    = 0.f;  // Pressure (AGC target 0..1 -> 0..0.98)
    x->exc_imp_shape  = 0.5f;  // Impulse-only Shape (old shape logic)
    x->exc_shape      = 0.5f;  // Noise Color (slope EQ)
    // Feedback-loop AGC defaults (0..1 mapped to time constants in DSP)
    x->fb_agc_attack  = 0.15f;
    x->fb_agc_release = 0.55f;
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




    // clear modulation matrix (bank 1 + bank 2)
    for(int i=0;i<JB_N_MODSRC;i++)
        for(int j=0;j<JB_N_MODTGT;j++){
            x->mod_matrix[i][j]  = 0.f;
            x->mod_matrix2[i][j] = 0.f;
        }
    x->basef0_ref=261.626f; // C4

    x->max_voices = JB_ATTACK_VOICES;
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
    x->fb_hp_a=0.f;

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
    // Spatial coupling controls (node/antinode; gain-level only)
    x->in_position      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("position_x"));    // excitation X pos 0..1
    x->in_positionY     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("position_y"));    // excitation Y pos 0..1
// excitation width 0..1
// 0..5
    x->in_pickupX       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("pickup_x"));       // pickup X pos 0..1
    x->in_pickupY       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("pickup_y"));       // pickup Y pos 0..1
    x->in_odd_even      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("odd_even"));      // -1..+1 (odd vs even emphasis)
// pickup width 0..1
// -1..+1 (closed..open)

// Individual

    // SPACE defaults (global room)
    x->space_size = 0.25f;
    x->space_decay = 0.35f;
    x->space_diffusion = 0.6f;
    x->space_damping = 0.25f;

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

    x->in_exc_density       = floatinlet_new(&x->x_obj, &x->exc_density);
    x->in_exc_imp_shape    = floatinlet_new(&x->x_obj, &x->exc_imp_shape);
    x->in_exc_shape         = floatinlet_new(&x->x_obj, &x->exc_shape);

    // Feedback AGC inlets (placed AFTER timbre/color, BEFORE LFO index)
    x->in_fb_agc_attack      = floatinlet_new(&x->x_obj, &x->fb_agc_attack);
    x->in_fb_agc_release     = floatinlet_new(&x->x_obj, &x->fb_agc_release);

    // LFO inlets
    // --- MOD SECTION (starts after exciter controls) ---
    x->in_lfo_index  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_index"));
    x->in_lfo_shape  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_shape"));
    x->in_lfo_rate   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_rate"));
    x->in_lfo_phase  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_phase"));
    x->in_lfo_amount = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float,  gensym("lfo_amount"));

    // Target selector inlets must accept bare selectors (e.g. a message box containing "damper_1"),
    // so we route them through proxies that implement an 'anything' handler.
    x->tgtproxy_lfo1 = (jb_tgtproxy *)pd_new(jb_tgtproxy_class);
    x->tgtproxy_lfo2 = (jb_tgtproxy *)pd_new(jb_tgtproxy_class);

    if (x->tgtproxy_lfo1){ x->tgtproxy_lfo1->owner = x; x->tgtproxy_lfo1->lane = 0; }
    if (x->tgtproxy_lfo2){ x->tgtproxy_lfo2->owner = x; x->tgtproxy_lfo2->lane = 1; }

    x->in_lfo1_target = inlet_new(&x->x_obj, x->tgtproxy_lfo1 ? &x->tgtproxy_lfo1->p_pd : &x->x_obj.ob_pd, 0, 0);
    x->in_lfo2_target = inlet_new(&x->x_obj, x->tgtproxy_lfo2 ? &x->tgtproxy_lfo2->p_pd : &x->x_obj.ob_pd, 0, 0);




    // Outs
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);
// checkpoint revert init

    x->out_index   = outlet_new(&x->x_obj, &s_float); // 1-based index reporter
    return (void *)x;
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
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position,      gensym("position"),      A_DEFFLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position_x,    gensym("position_x"),    A_DEFFLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position_y,    gensym("position_y"),    A_DEFFLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pickupL,      gensym("pickup_x"),       A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pickupR,      gensym("pickup_y"),       A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_odd_even,     gensym("odd_even"),       A_DEFFLOAT, 0);
// legacy alias (sets both X and Y)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pickup,       gensym("pickup"),        A_DEFFLOAT, 0);
// LFO + ADSR methods
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_shape, gensym("lfo_shape"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_rate,  gensym("lfo_rate"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_phase, gensym("lfo_phase"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_index, gensym("lfo_index"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_amount, gensym("lfo_amount"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo1_target, gensym("lfo1_target"), A_SYMBOL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo2_target, gensym("lfo2_target"), A_SYMBOL, 0);
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
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index_forward, gensym("forward"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index_backward, gensym("backward"), 0);

    
    // offline render (testing/regression)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_render, gensym("render"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_renderwrite, gensym("renderwrite"), A_SYMBOL, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_renderclear, gensym("renderclear"), 0);

}

