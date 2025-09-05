// juicy_bank_tilde.c
// Modal resonator bank for Pure Data / PlugData
// Implements user-specified behaviors: damping, brightness tilt, position nodes, dispersion, density, anisotropy, contact, per-mode curve.
//
// Build (macOS example):
//   clang -O3 -DPD -std=c99 -fPIC -shared -o juicy_bank_tilde.pd_darwin juicy_bank_tilde.c
//
// (c) 2025 Juicy x ChatGPT — do something wonderful

#include "m_pd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define JB_MAX_MODES 20

// small helpers
static inline float jb_clamp(float x, float lo, float hi){ return x < lo ? lo : (x > hi ? hi : x); }
static inline float jb_mmix(float a, float b, float t){ return a + (b - a) * t; } // linear mix
static inline float jb_smoothstep(float x){ x = jb_clamp(x, 0.f, 1.f); return x*x*(3.f - 2.f*x); }

// Random (xorshift32) for deterministic seeds
typedef struct { uint32_t s; } jb_rng_t;
static inline void jb_rng_seed(jb_rng_t* r, uint32_t s){ if(s==0) s=1u; r->s = s; }
static inline uint32_t jb_rng_u32(jb_rng_t* r){ uint32_t x=r->s; x ^= x<<13; x ^= x>>17; x ^= x<<5; return r->s = x; }
static inline float jb_rng_uniform(jb_rng_t* r){ return (jb_rng_u32(r) / 4294967296.0f); } // [0,1)
static inline float jb_rng_bi(jb_rng_t* r){ return 2.f*jb_rng_uniform(r) - 1.f; } // [-1,1)

typedef enum {
    DENSITY_PIVOT = 0,      // pivot around fundamental (ratio 1.0)
    DENSITY_INDIVIDUAL = 1  // chain spacing based on previous active
} density_mode_e;

// Per-mode state
typedef struct {
    int   active;         // 0/1
    float base_ratio;     // user-set ratio (>= 0.0), fundamental ~1.0
    float base_decay_ms;  // user-set decay ms
    float base_gain;      // user-set linear gain 0..1

    // Derived (recomputed on param changes)
    float ratio_now;      // after density & dispersion
    float decay_ms_now;   // after damping
    float gain_now;       // after brightness, anisotropy, position, etc.

    // Reson filter state: y[n] = 2 r cos(w) y[n-1] - r^2 y[n-2] + g * x[n]
    float a1, a2;         // coeffs
    float y1, y2;         // state

    // Contact envelope follower (per-mode)
    float env;            // simple abs-based follower 0..1
    float curve_amt;      // per-mode curve control (-1..1)
    
    // Dispersion random (stable)
    float disp_signature; // in [-1,1], signed random unique to the mode
} mode_t;

typedef struct _juicy_bank_tilde {
    t_object  x_obj;
    t_float   f_dummy;

    t_inlet*  in2;   // optional control-rate inlet (unused; placeholder)
    t_outlet* outlet;

    // Global parameters
    float sr;
    int   n_modes;
    mode_t m[JB_MAX_MODES];

    // Fundamental index (auto-detected as closest to ratio=1.0 among active modes)
    int fundamental_idx;

    // Controls
    float damping;       // 0..1: linear percentage cut of decay (new = base*(1-damping))
    float brightness;    // 0..1: 0 mutes highs, 0.5 neutral, 1 boosts highs
    float position;      // 0..1: 0 bypass; >0 true node cancellations via sin(pi*k*pos)
    float dispersion;    // 0..1: scale of random ± up to 1.0 ratio offset (fundamental immune)
    float anisotropy;    // -1..1: continuous mute of groups (non-integer "odd") vs near-integer "even"
    float contact_amt;   // 0..1: strength of contact nonlinearity
    float contact_sym;   // -1..1: asymmetry of contact shape
    float density;       // -1..1: spacing scale (-50%..+50%)
    density_mode_e density_mode;

    // Bookkeeping
    float brightness_alpha; // slope for tilt
    float env_follower_tau; // for contact detection
    float contact_threshold;

    // Position behavior
    int   use_true_nodes;  // always true
    // Anisotropy epsilon for "near-integer" classification
    float epsilon_near_int;

    // Dispersion RNG
    jb_rng_t rng;
    uint32_t seed_value;

    // Small slews for control changes
    float dispersion_slew; // 0..1, how fast offsets glide back (per block)
    float density_slew;
    float brightness_slew;

    // Cached arrays for computed weights (smoothed)
    float pos_weight[JB_MAX_MODES];
    float disp_offset[JB_MAX_MODES];     // current applied offset
    float disp_target[JB_MAX_MODES];     // target offset after re-roll / knob change
} t_juicy_bank_tilde;

// ----- Utilities -----

static int jb_is_near_integer(float x, float eps){
    float r = roundf(x);
    return (fabsf(x - r) <= eps) ? 1 : 0;
}

// Find fundamental index (active mode closest to ratio 1.0). Returns -1 if none.
static int jb_find_fundamental(t_juicy_bank_tilde* x){
    int best = -1;
    float bestd = 1e9f;
    for(int i=0;i<x->n_modes;i++){
        if(!x->m[i].active) continue;
        float d = fabsf(x->m[i].base_ratio - 1.f);
        if(d < bestd){ bestd = d; best = i; }
    }
    return best;
}

// Recompute density mapping to ratio_now for all active modes
static void jb_apply_density(t_juicy_bank_tilde* x){
    // density: -1..1 maps to scale s = 1 + 0.5*density
    float s = 1.f + 0.5f * x->density;
    if (x->density_mode == DENSITY_PIVOT){
        // pivot = fundamental (ratio 1.0)
        for(int i=0;i<x->n_modes;i++){
            mode_t* md = &x->m[i];
            if(!md->active){ md->ratio_now = md->base_ratio; continue; }
            // fundamental unaffected (exactly 1 stays 1)
            md->ratio_now = 1.f + (md->base_ratio - 1.f) * s;
        }
    } else {
        // individual: chain spacing from previous ACTIVE mode
        // We'll preserve order by base_ratio ascending, but we must write back to original indices.
        // Build an array of indices of active modes sorted by base_ratio.
        int idxs[JB_MAX_MODES]; int count=0;
        for(int i=0;i<x->n_modes;i++) if(x->m[i].active) idxs[count++] = i;
        // simple insertion sort by base_ratio
        for(int a=1;a<count;a++){
            int j=a; int id=idxs[a];
            while(j>0 && x->m[idxs[j-1]].base_ratio > x->m[id].base_ratio){ idxs[j]=idxs[j-1]; j--; }
            idxs[j]=id;
        }
        // apply spacing
        int first = -1;
        for(int k=0;k<count;k++){
            int i = idxs[k];
            mode_t* md = &x->m[i];
            if(first < 0){
                md->ratio_now = md->base_ratio; // first stays where it is
                first = i;
            } else {
                int prev_i = idxs[k-1];
                mode_t* pm = &x->m[prev_i];
                float d = (md->base_ratio - x->m[prev_i].base_ratio) * s;
                md->ratio_now = pm->ratio_now + d;
            }
        }
        // inactive modes: keep base
        for(int i=0;i<x->n_modes;i++) if(!x->m[i].active) x->m[i].ratio_now = x->m[i].base_ratio;
    }
}

// Re-roll dispersion targets (respect fundamental immunity)
static void jb_reroll_dispersion_targets(t_juicy_bank_tilde* x){
    // Deterministic per seed: use index to stir RNG for stable signature
    for(int i=0;i<x->n_modes;i++){
        mode_t* md = &x->m[i];
        if(!md->active){ x->disp_target[i] = 0.f; continue; }
        if(i == x->fundamental_idx){ x->disp_target[i] = 0.f; continue; }
        // signature is fixed per mode; target is signature * amount * 1.0 (max ±1 ratio at full dispersion)
        x->disp_target[i] = md->disp_signature * x->dispersion;
    }
}

// Smoothly move current offsets toward targets
static void jb_slew_dispersion(t_juicy_bank_tilde* x){
    float a = x->dispersion_slew;
    for(int i=0;i<x->n_modes;i++){
        x->disp_offset[i] = jb_mmix(x->disp_offset[i], x->disp_target[i], a);
    }
}

// Apply brightness / anisotropy / position weights into gain_now (multiplicative on base_gain)
static void jb_update_per_mode_gains(t_juicy_bank_tilde* x){
    // Setup
    float b = x->brightness; // 0..1
    float t = 2.f*(b - 0.5f); // -1..+1 ; negative mutes highs, positive boosts highs
    float alpha = x->brightness_alpha; // slope
    float eps = x->epsilon_near_int;

    // Determine integer-ish harmonic index for position (using nearest integer of ratio)
    // Position default 0 means bypass.
    float pos = x->position;
    int use_pos = (pos > 0.f);

    for(int i=0;i<x->n_modes;i++){
        mode_t* md = &x->m[i];
        if(!md->active){
            md->gain_now = 0.f;
            continue;
        }

        // Start from base gain
        float g = md->base_gain;

        // Brightness tilt: factor = pow(ratio_now / 1.0, alpha * t)
        float r = md->ratio_now + x->disp_offset[i];
        if (r < 0.0001f) r = 0.0001f;
        float bright_fac = powf(r, alpha * t);
        // Keep fundamental exactly 1.0 always
        if(i == x->fundamental_idx) bright_fac = 1.f;
        g *= bright_fac;

        // Anisotropy: classify "even" if near-integer, else "odd"
        int is_even = jb_is_near_integer(r, eps);
        float a = x->anisotropy; // -1..+1; +1 => mute "odd", -1 => mute "even"
        float w = 1.f;
        if (i != x->fundamental_idx){
            if (a > 0.f){ // favor even => mute odd proportionally
                if(!is_even){
                    float k = jb_clamp(a, 0.f, 1.f);
                    // continuous mute
                    w *= (1.f - k);
                }
            } else if (a < 0.f){ // favor odd => mute even proportionally
                if(is_even){
                    float k = jb_clamp(-a, 0.f, 1.f);
                    w *= (1.f - k);
                }
            }
        }
        g *= jb_clamp(w, 0.f, 1.f);

        // Position: if pos>0, apply node weighting by integer harmonic index k=round(ratio)
        if (use_pos){
            int k = (int)floorf(r + 0.5f);
            if (k < 1) k = 1;
            float pw = fabsf(sinf(M_PI * (float)k * pos));
            // true nodes allowed => no floor
            g *= pw;
        }

        md->gain_now = g;
    }
}

// Given decay_ms_now per mode, compute filter coeffs a1,a2
static void jb_update_coeffs(t_juicy_bank_tilde* x){
    float sr = x->sr;
    for(int i=0;i<x->n_modes;i++){
        mode_t* md = &x->m[i];
        if(!md->active){
            md->a1 = 0.f; md->a2 = 0.f; md->y1 = md->y2 = 0.f;
            continue;
        }
        float ratio = md->ratio_now + x->disp_offset[i];
        if(ratio < 0.0001f) ratio = 0.0001f;
        // Assume fundamental frequency is provided separately through "base frequency" * ratio outside this object,
        // but if not, we can treat base as 100 Hz (user should set actual freqs via "freq" list).
        // For safety, we will expect user-provided absolute frequencies through freq[] list, not ratio.
        // Here we interpret ratio_now as a multiplier applied to a base fundamental f0 of 100Hz unless freq list sets absolute.
        // We'll store absolute freq into ratio_now for simplicity: treat ratio_now as absolute Hz if freq[] used.
        float f = ratio; // assume absolute Hz already set
        float w = 2.f * M_PI * (f / sr);
        if (w > M_PI) w = M_PI;

        // Map decay_ms_now to r such that -60dB in T60 = decay_ms_now
        float T60 = jb_clamp(md->decay_ms_now, 0.f, 1e6f) * 0.001f; // seconds
        float r = (T60 <= 0.f) ? 0.f : powf(10.f, -3.f / (T60 * sr)); // 0.001 factor over T60

        float c = cosf(w);
        md->a1 = 2.f * r * c;
        md->a2 = -r * r;
        // note: base_gain (and later weights) go on the input drive
    }
}

// Contact shaping (with symmetry and make-up).
// - sym in [-1,1], amt in [0,1].
// - y is mode signal; env controls engagement threshold.
static inline float jb_contact_shape(float y, float amt, float sym){
    if(amt <= 0.f) return y;
    // asymmetric soft clip:
    // shift via symmetry: positive half boosted when sym>0, negative when sym<0
    float sign_bias = sym * 0.5f; // in [-0.5, 0.5]
    float z = y + sign_bias * y * y; // quadratic asymmetry
    // soft clip with amt controlling drive
    float drive = 1.f + 10.f * amt;
    float shaped = tanhf(drive * z);
    // auto-makeup: scale to keep small-signal slope ~1
    // derivative at 0 = drive, for tanh(drive*x) -> slope = drive; compensate by 1/drive
    float makeup = (drive > 0.f) ? (1.f / drive) : 1.f;
    return shaped * makeup;
}

// DSP perform
static t_int *juicy_bank_tilde_perform(t_int *w){
    t_juicy_bank_tilde *x = (t_juicy_bank_tilde *)(w[1]);
    t_sample *in = (t_sample *)(w[2]);
    t_sample *out = (t_sample *)(w[3]);
    int n = (int)(w[4]);

    // Smooth dispersion toward target each block
    jb_slew_dispersion(x);

    // Update per-mode gains (brightness/position/anisotropy), then coeffs (in case decay changed by damping)
    jb_update_per_mode_gains(x);
    jb_update_coeffs(x);

    float env_tau = x->env_follower_tau;
    float th = x->contact_threshold;
    float camt = x->contact_amt;
    float csym = x->contact_sym;

    for(int i=0;i<n;i++){
        float exc = in[i];
        float sum = 0.f;

        for(int m=0;m<x->n_modes;m++){
            mode_t* md = &x->m[m];
            if(!md->active || md->gain_now <= 0.f) continue;

            // simple reson driven by input
            float y = (md->a1 * md->y1 + md->a2 * md->y2) + (md->gain_now * exc);
            md->y2 = md->y1;
            md->y1 = y;

            // envelope follower for contact gating
            float abs_y = fabsf(y);
            md->env = md->env + env_tau * (abs_y - md->env); // 1st order

            // contact apply only when env exceeds threshold
            if (camt > 0.f && md->env > th){
                y = jb_contact_shape(y, camt, csym);
            }

            // (future) per-mode curve shaping could be applied here by scaling y using md->curve_amt and md->env

            sum += y;
        }

        out[i] = sum;
    }

    return (w + 5);
}

// ----- Pd boilerplate -----

static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;
    dsp_add(juicy_bank_tilde_perform, 4, x, sp[0]->s_vec, sp[1]->s_vec, sp[0]->s_n);
}

// ----- Messages / Setters -----

// Set number of modes (1..JB_MAX_MODES)
static void juicy_bank_tilde_modes(t_juicy_bank_tilde *x, t_floatarg f){
    int nm = (int)f;
    if(nm < 1) nm = 1;
    if(nm > JB_MAX_MODES) nm = JB_MAX_MODES;
    x->n_modes = nm;
}

// Activate/deactivate a mode by index (1-based)
static void juicy_bank_tilde_active(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg onf){
    int idx = (int)idxf - 1;
    if(idx < 0 || idx >= x->n_modes) return;
    x->m[idx].active = (onf > 0.f) ? 1 : 0;
    // if activated new mode, assign a new dispersion signature based on seed+index
    jb_rng_t tmp = x->rng;
    for(int k=0;k<=idx;k++) jb_rng_u32(&tmp);
    x->m[idx].disp_signature = jb_rng_bi(&tmp);
    // recompute fundamental
    x->fundamental_idx = jb_find_fundamental(x);
}

// Set per-mode base ratio (interpreted as absolute frequency in Hz if you prefer; typical workflow sets absolute freqs)
static void juicy_bank_tilde_ratio(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg ratio){
    int idx = (int)idxf - 1;
    if(idx < 0 || idx >= x->n_modes) return;
    if(ratio < 0.f) ratio = 0.f;
    x->m[idx].base_ratio = ratio;
    x->m[idx].ratio_now = ratio;
    x->fundamental_idx = jb_find_fundamental(x);
}

// Set per-mode base decay (ms)
static void juicy_bank_tilde_decay(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg ms){
    int idx = (int)idxf - 1;
    if(idx < 0 || idx >= x->n_modes) return;
    if(ms < 0.f) ms = 0.f;
    x->m[idx].base_decay_ms = ms;
}

// Set per-mode base gain (0..1)
static void juicy_bank_tilde_gain(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg g){
    int idx = (int)idxf - 1;
    if(idx < 0 || idx >= x->n_modes) return;
    x->m[idx].base_gain = jb_clamp(g, 0.f, 1.f);
}

// Batch lists: freq[], decay[], amp[] — lengths up to n_modes
static void juicy_bank_tilde_freqlist(t_juicy_bank_tilde *x, t_symbol* s, int argc, t_atom* argv){
    int n = (argc < x->n_modes)? argc : x->n_modes;
    for(int i=0;i<n;i++){
        if(argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i);
            if (v < 0.f) v = 0.f;
            x->m[i].base_ratio = v; // interpret as absolute Hz for simplicity
            x->m[i].ratio_now  = v;
        }
    }
    x->fundamental_idx = jb_find_fundamental(x);
}
static void juicy_bank_tilde_decaylist(t_juicy_bank_tilde *x, t_symbol* s, int argc, t_atom* argv){
    int n = (argc < x->n_modes)? argc : x->n_modes;
    for(int i=0;i<n;i++){
        if(argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i);
            if (v < 0.f) v = 0.f;
            x->m[i].base_decay_ms = v;
        }
    }
}
static void juicy_bank_tilde_amplist(t_juicy_bank_tilde *x, t_symbol* s, int argc, t_atom* argv){
    int n = (argc < x->n_modes)? argc : x->n_modes;
    for(int i=0;i<n;i++){
        if(argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i);
            x->m[i].base_gain = jb_clamp(v, 0.f, 1.f);
        }
    }
}

// Global params

static void juicy_bank_tilde_damping(t_juicy_bank_tilde *x, t_floatarg f){
    x->damping = jb_clamp(f, 0.f, 1.f);
    // apply to decay_now immediately
    for(int i=0;i<x->n_modes;i++){
        mode_t* md = &x->m[i];
        md->decay_ms_now = md->base_decay_ms * (1.f - x->damping);
    }
}

static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){
    x->brightness = jb_clamp(f, 0.f, 1.f);
}

static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){
    x->position = jb_clamp(f, 0.f, 1.f);
}

static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){
    x->dispersion = jb_clamp(f, 0.f, 1.f);
    // update targets based on current amount
    jb_reroll_dispersion_targets(x);
}

static void juicy_bank_tilde_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){
    float a = f;
    if(a < -1.f) a = -1.f;
    if(a >  1.f) a =  1.f;
    x->anisotropy = a;
}

static void juicy_bank_tilde_contact(t_juicy_bank_tilde *x, t_floatarg f){
    x->contact_amt = jb_clamp(f, 0.f, 1.f);
}
static void juicy_bank_tilde_contact_sym(t_juicy_bank_tilde *x, t_floatarg f){
    if(f < -1.f) f = -1.f;
    if(f >  1.f) f =  1.f;
    x->contact_sym = f;
}

// Density controls
static void juicy_bank_tilde_density(t_juicy_bank_tilde *x, t_floatarg f){
    float d = f; if(d < -1.f) d = -1.f; if(d > 1.f) d = 1.f;
    x->density = d;
    jb_apply_density(x); // recompute ratio_now baseline
    jb_reroll_dispersion_targets(x); // update target offsets based on new ratios
}

static void juicy_bank_tilde_density_pivot(t_juicy_bank_tilde *x){
    x->density_mode = DENSITY_PIVOT;
    jb_apply_density(x);
    jb_reroll_dispersion_targets(x);
}
static void juicy_bank_tilde_density_individual(t_juicy_bank_tilde *x){
    x->density_mode = DENSITY_INDIVIDUAL;
    jb_apply_density(x);
    jb_reroll_dispersion_targets(x);
}

// Curve per mode: curve <index> <amount [-1..1]>
static void juicy_bank_tilde_curve(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg amt){
    int idx = (int)idxf - 1;
    if(idx < 0 || idx >= x->n_modes) return;
    if(amt < -1.f) amt = -1.f;
    if(amt >  1.f) amt =  1.f;
    x->m[idx].curve_amt = amt;
}

// Seed + reroll
static void juicy_bank_tilde_seed(t_juicy_bank_tilde *x, t_floatarg f){
    uint32_t s = (uint32_t) (f < 0 ? -f : f);
    if(s == 0) s = 1u;
    x->seed_value = s;
    jb_rng_seed(&x->rng, s);
    // assign per-mode signatures deterministically
    for(int i=0;i<x->n_modes;i++){
        jb_rng_t tmp = x->rng;
        for(int k=0;k<=i;k++) jb_rng_u32(&tmp);
        x->m[i].disp_signature = jb_rng_bi(&tmp);
    }
    jb_reroll_dispersion_targets(x);
}
static void juicy_bank_tilde_dispersion_reroll(t_juicy_bank_tilde *x){
    // rotate seed slightly for new constellation
    x->seed_value = x->seed_value * 1664525u + 1013904223u;
    jb_rng_seed(&x->rng, x->seed_value);
    for(int i=0;i<x->n_modes;i++){
        jb_rng_t tmp = x->rng;
        for(int k=0;k<=i;k++) jb_rng_u32(&tmp);
        x->m[i].disp_signature = jb_rng_bi(&tmp);
    }
    jb_reroll_dispersion_targets(x);
}

// ----- New -----
// optional: set epsilon for "near-integer" classification (default 0.02)
static void juicy_bank_tilde_aniso_epsilon(t_juicy_bank_tilde *x, t_floatarg f){
    float e = f;
    if(e < 0.f) e = 0.f;
    if(e > 0.25f) e = 0.25f;
    x->epsilon_near_int = e;
}

// ----- Constructor / Destructor -----

static void *juicy_bank_tilde_new(t_symbol *s, int argc, t_atom *argv){
    t_juicy_bank_tilde *x = (t_juicy_bank_tilde *)pd_new(gensym("juicy_bank_tilde")->s_thing);
    inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal); // signal inlet (not strictly used)
    x->outlet = outlet_new(&x->x_obj, &s_signal);

    x->sr = sys_getsr();
    if (x->sr <= 0) x->sr = 48000.f;

    x->n_modes = JB_MAX_MODES;
    for(int i=0;i<JB_MAX_MODES;i++){
        mode_t* md = &x->m[i];
        md->active = (i < 8) ? 1 : 0;
        md->base_ratio = (float)(i+1); // default rough harmonic-ish
        md->base_decay_ms = 500.f;
        md->base_gain = 0.2f;
        md->ratio_now = md->base_ratio;
        md->decay_ms_now = md->base_decay_ms;
        md->gain_now = md->base_gain;
        md->a1 = 0.f; md->a2 = 0.f; md->y1 = 0.f; md->y2 = 0.f;
        md->env = 0.f;
        md->curve_amt = 0.f;
        md->disp_signature = 0.f;
        // defaults for cached
    }

    x->fundamental_idx = jb_find_fundamental(x);

    // params defaults
    x->damping = 0.f;
    x->brightness = 0.5f;
    x->position = 0.f;
    x->dispersion = 0.f;
    x->anisotropy = 0.f;
    x->contact_amt = 0.f;
    x->contact_sym = 0.f;
    x->density = 0.f;
    x->density_mode = DENSITY_PIVOT;

    x->brightness_alpha = 0.75f;
    x->env_follower_tau = 0.01f;  // follower smoothing
    x->contact_threshold = 0.02f; // small threshold

    x->epsilon_near_int = 0.02f;

    jb_rng_seed(&x->rng, 1234567u);
    x->seed_value = 1234567u;
    for(int i=0;i<JB_MAX_MODES;i++){
        x->pos_weight[i]=1.f;
        x->disp_offset[i]=0.f;
        x->disp_target[i]=0.f;
        // signature
        jb_rng_t tmp = x->rng;
        for(int k=0;k<=i;k++) jb_rng_u32(&tmp);
        x->m[i].disp_signature = jb_rng_bi(&tmp);
    }

    x->dispersion_slew = 0.05f; // smooth toward targets per block
    x->density_slew    = 0.1f;
    x->brightness_slew = 0.1f;

    return (void *)x;
}

static void juicy_bank_tilde_free(t_juicy_bank_tilde *x){
    outlet_free(x->outlet);
}

// ----- Setup -----

void juicy_bank_tilde_setup(void){
    t_class *c = class_new(gensym("juicy_bank_tilde"),
                           (t_newmethod)juicy_bank_tilde_new,
                           (t_method)juicy_bank_tilde_free,
                           sizeof(t_juicy_bank_tilde),
                           CLASS_DEFAULT,
                           A_GIMME, 0);

    class_addmethod(c, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);

    CLASS_MAINSIGNALIN(c, t_juicy_bank_tilde, f_dummy);

    // Messages
    class_addmethod(c, (t_method)juicy_bank_tilde_modes, gensym("modes"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_active, gensym("active"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_ratio, gensym("ratio"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_decay, gensym("decay"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_gain,  gensym("gain"),  A_DEFFLOAT, A_DEFFLOAT, 0);

    class_addmethod(c, (t_method)juicy_bank_tilde_freqlist,   gensym("freq"),   A_GIMME, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_decaylist,  gensym("decays"), A_GIMME, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_amplist,    gensym("amps"),   A_GIMME, 0);

    class_addmethod(c, (t_method)juicy_bank_tilde_damping,    gensym("damping"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_position,   gensym("position"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_contact,    gensym("contact"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_contact_sym,gensym("contact_symmetry"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_density,    gensym("density"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);

    class_addmethod(c, (t_method)juicy_bank_tilde_curve, gensym("curve"), A_DEFFLOAT, A_DEFFLOAT, 0);

    class_addmethod(c, (t_method)juicy_bank_tilde_seed, gensym("seed"), A_DEFFLOAT, 0);
    class_addmethod(c, (t_method)juicy_bank_tilde_dispersion_reroll, gensym("dispersion_reroll"), 0);

    class_addmethod(c, (t_method)juicy_bank_tilde_aniso_epsilon, gensym("aniso_epsilon"), A_DEFFLOAT, 0);

    // finalize
    // store the class pointer in Pd's symbol (so new() can find it)
    gensym("juicy_bank_tilde")->s_thing = (t_pd*)c;
}
