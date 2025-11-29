// juicy_bank~ — modal resonator bank (V5.0)
// 4-voice poly, true stereo banks, Behavior + Body + Individual inlets.
// NEW (V5.0):
//   • **Spacing** inlet (after dispersion, before anisotropy): nudges each mode toward the *next* harmonic
//     ratio (ceil or +1 if already integer). 0 = no shift, 1 = fully at next ratio.
//   • **64 modes by default**: startup ratios 1..64, gain=1.0, decay=1000 ms, attack=0, curve=0 (linear).
//   • **Resonant loudness normalization**: per-mode drive is scaled by (1 - 2 r cos(w) + r^2) so low freqs
//     are not inherently louder than highs for a fixed T60. This fixes the historical low-end bias
//     without artificially forcing per-mode gains.
//
// Build (macOS):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//     -I"/Applications/Pd-0.56-1.app/Contents/Resources/src" \
//     -arch arm64 -arch x86_64 -mmacosx-version-min=10.13 \
//     -bundle -undefined dynamic_lookup \
//     -o juicy_bank~.pd_darwin juicy_bank_tilde.c
//
// Build (Linux):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//     -I"/usr/include/pd" -shared -fPIC -Wl,-export-dynamic -lm \
//     -o juicy_bank~.pd_linux juicy_bank_tilde.c

#include "m_pd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ---------- limits ----------
#define JB_MAX_DECAY_S 5.0f  /* global opposite-damping ceiling in seconds */

#define JB_MAX_MODES    64
#define JB_MAX_VOICES    4
#define JB_FB_MAX      4096
#define JB_N_MODSRC    5
#define JB_N_MODTGT    15
#define JB_N_LFO       2
#define JB_PITCH_MOD_SEMITONES  2.0f

// ---------- utils ----------
static inline float jb_clamp(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }
static inline float jb_wrap01(float x){
    x = x - floorf(x);
    if (x < 0.f) x += 1.f;
    return x;
}
typedef struct { unsigned int s; } jb_rng_t;
static inline void jb_rng_seed(jb_rng_t *r, unsigned int s){ if(!s) s=1; r->s = s; }
static inline unsigned int jb_rng_u32(jb_rng_t *r){ unsigned int x = r->s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; r->s = x; return x; }
static inline float jb_rng_uni(jb_rng_t *r){ return (jb_rng_u32(r) >> 8) * (1.0f/16777216.0f); }
static inline float jb_rng_bi(jb_rng_t *r){ return 2.f * jb_rng_uni(r) - 1.f; }

static int jb_is_near_integer(float x, float eps){ float n=roundf(x); return fabsf(x-n)<=eps; }
static inline float jb_midi_to_hz(float n){ return 440.f * powf(2.f, (n-69.f)/12.f); }

typedef enum { DENSITY_PIVOT=0, DENSITY_INDIV=1 } jb_density_mode;

typedef struct {
    // base params (shared template per mode)
    float base_ratio, base_decay_ms, base_gain;
    float attack_ms, curve_amt, pan;
    int   active;
    int   keytrack; // 1 = track f0 (ratio), 0 = absolute Hz

    // signatures (random)
    float disp_signature;
    float micro_sig;
} jb_mode_base_t;

typedef struct {
    
    // release envelope (global per-voice)
    float rel_env;
    // runtime per-mode
    float ratio_now, decay_ms_now, gain_now;
    float t60_s, decay_u;

    // per-ear per-hit randomizations
    float md_hit_offsetL, md_hit_offsetR;   // micro detune offsets
    float bw_hit_ratioL, bw_hit_ratioR;     // twin detune ratios

    // LEFT states
    float a1L,a2L, y1L,y2L, a1bL,a2bL, y1bL,y2bL, envL, y_pre_lastL, normL;
    // RIGHT states
    float a1R,a2R, y1R,y2R, a1bR,a2bR, y1bR,y2bR, envR, y_pre_lastR, normR;

    // drive/hit
    float driveL, driveR;
    int   hit_gateL, hit_coolL, hit_gateR, hit_coolR;
    int   nyq_kill;
} jb_mode_rt_t;

typedef enum { V_IDLE=0, V_HELD=1, V_RELEASE=2 } jb_vstate;

typedef struct {
    jb_vstate state;
    float f0, vel, energy;

    // projected behavior (per voice)
    float pitch_x;
    float brightness_v;
    float bandwidth_v;
    float decay_pitch_mul;
    float decay_vel_mul;
    float stiffen_add;

    // sympathetic multipliers
    float cr_gain_mul[JB_MAX_MODES];
    float cr_decay_mul[JB_MAX_MODES];

    // dispersion morph targets
    float disp_offset[JB_MAX_MODES];
    float disp_target[JB_MAX_MODES];

    
    // release envelope (global per-voice)
    float rel_env;
    // runtime per-mode
    jb_mode_rt_t m[JB_MAX_MODES];

    // --- FEEDBACK per-voice, per-ear states ---
// DC HP state (30 Hz highpass on voice sum before feedback path)
    float fb_hp_x1L, fb_hp_y1L, fb_hp_x1R, fb_hp_y1R;
// 2-sample delay registers (core feedback delay line)
    float fb_d1L, fb_d2L, fb_d1R, fb_d2R;
// one-pole lowpass state in the feedback path (per-ear)
    float fb_lpL, fb_lpR;
// smoothed pulse state (per-ear)
    float fb_pulseL, fb_pulseR;
// previous-block filtered+delayed buffers (legacy; currently unused but kept for compatibility)
    float fb_prevL[JB_FB_MAX];
    float fb_prevR[JB_FB_MAX];
    int   fb_prev_len;
} jb_voice_t;

// ---------- the object ----------
static t_class *juicy_bank_tilde_class;

typedef struct _juicy_bank_tilde {
    t_object  x_obj; t_float f_dummy; t_float sr;

    int n_modes;
    int active_modes;              // number of currently active partials (0..n_modes)
    t_inlet *in_partials;          // message inlet for 'partials' (float 0..n_modes)
    t_outlet *out_index;           // float outlet reporting current selected partial (1-based)
    jb_mode_base_t base[JB_MAX_MODES];

    // BODY globals
    float release_amt;

    
    // --- STRETCH (message-only, -1..1; internal musical scaling) ---
    float stretch;

    float warp; // -1..+1 bias for stretch distribution
float damping, brightness, position; float damp_broad, damp_point;
    float density_amt; jb_density_mode density_mode;
    float dispersion, dispersion_last;
    float offset_amt;
    float aniso, aniso_eps;

    // realism/misc
    float phase_rand; int phase_debug;
    float bandwidth;        // base for Bloom
    float micro_detune;     // base for micro detune
    float basef0_ref;

    // FEEDBACK params (global controls)
    t_inlet *in_fb_drive; // float inlet (0..1)
    t_inlet *in_fb_amt;   // float inlet (-1..+1)
    t_inlet *in_fb_timer; // float inlet (0..1)
    t_inlet *in_fb_fmax;  // float inlet (0..1)
    float fb_hp_a;        // fixed 30 Hz 1-pole HP coeff
    float fb_amt;         // -1..+1, startup 0
    float fb_amt_z;       // slewed value
    float fb_slew_a;      // slew pole (~10 ms)
    float fb_drive;       // 0..1
    float fb_timer;       // 0..1 crossfade between raw feedback and pulse-shaped feedback
    float fb_fmax;        // 0..1 maps to max pulse frequency / LP cutoff


    // BEHAVIOR depths
    float stiffen_amt, shortscale_amt, linger_amt, tilt_amt, bite_amt, bloom_amt, crossring_amt;

    // voices
    int   max_voices;
    jb_voice_t v[JB_MAX_VOICES];

    // current edit index for Individual setters
    int edit_idx;

    // RNG
    jb_rng_t rng;

    // DC HP
    float hp_a, hpL_x1, hpL_y1, hpR_x1, hpR_y1;

    // IO
    // main stereo exciter inputs
    t_inlet *inR;
    // per-voice exciter inputs (optional)
    t_inlet *in_vL[JB_MAX_VOICES];
    t_inlet *in_vR[JB_MAX_VOICES];
    int exciter_mode; // 0 = use main L/R (legacy), 1 = use per-voice pairs

    t_outlet *outL, *outR;

    // INLET pointers
    // Behavior (reduced)
    t_inlet *in_crossring;
    
        t_inlet *in_release;
// Body controls (damping, brightness, position, density, dispersion, anisotropy)
    t_inlet *in_damping, *in_damp_broad, *in_damp_point, *in_brightness, *in_position, *in_density, *in_stretch, *in_warp, *in_dispersion, *in_offset, *in_aniso;
    // Individual
    t_inlet *in_index, *in_ratio, *in_gain, *in_attack, *in_decay, *in_curve, *in_pan, *in_keytrack;

    // --- SINE (AM across modes) ---
    float sine_pitch;   // 0..1
    float sine_depth;   // 0..1
    float sine_phase;   // 0..1 (wraps)

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

    float adsr_ms;     // ADSR envelope (0..1) from exciter (legacy name)

    // inlets for SINE
    t_inlet *in_sine_pitch;
    t_inlet *in_sine_depth;
    t_inlet *in_sine_phase;

    // --- LFO + ADSR inlets (for modulation matrix) ---
    t_inlet *in_lfo_shape;
    t_inlet *in_lfo_rate;
    t_inlet *in_lfo_phase;
    t_inlet *in_lfo_index;
    t_inlet *in_matrix;   // inlet for modulation-matrix messages (anything)
    t_inlet *in_adsr_ms;
// --- SNAPSHOT (bake) undo buffer ---
int _undo_valid;
float _undo_base_gain[JB_MAX_MODES];
float _undo_base_decay_ms[JB_MAX_MODES];
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
        if (shape > 4) shape = 4;

        float val = 0.f;
        if (shape == 1){
            // saw: -1..+1
            val = 2.f * ph - 1.f;
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


// ---------- helpers ----------
static float jb_bright_gain(float ratio_rel, float b){
    float t=(jb_clamp(b,0.f,1.f)-0.5f)*2.f; float p=0.6f*t; float rr=jb_clamp(ratio_rel,1.f,1e6f);
    return powf(rr, p);
}
static float jb_position_weight(float ratio_rel, float pos){
    if (pos<=0.f) return 1.f;
    float k = roundf(jb_clamp(ratio_rel,1.f,1e6f));
    return fabsf(sinf((float)M_PI * k * jb_clamp(pos,0.f,1.f)));
}
static inline float jb_curve_shape_gain(float u, float curve){
    if (u <= 0.f) return 1.f;
    if (u >= 1.f) return 1.f;
    float gamma;
    if (curve < 0.f){ float t = -curve; gamma = 1.f - t*(1.f - 0.35f); }
    else if (curve > 0.f){ float t = curve;  gamma = 1.f + t*(3.0f - 1.f); }
    else return 1.f;
    float phi = powf(jb_clamp(u,0.f,1.f), gamma);
    float delta = phi - u;
    return powf(10.f, -3.f * delta);
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

// ---------- behavior projection ----------
static void jb_project_behavior_into_voice(t_juicy_bank_tilde *x, jb_voice_t *v){
    float xfac = (x->basef0_ref>0.f)? (v->f0 / x->basef0_ref) : 1.f;
    if (xfac < 1e-6f) xfac = 1e-6f;
    v->pitch_x = xfac;

    // Stiffen → extra dispersion depth
    float k_disp = (0.02f + 0.10f * jb_clamp(x->stiffen_amt,0.f,1.f));
    float alpha  = 0.60f + 0.20f * x->stiffen_amt;
    v->stiffen_add = k_disp * powf(xfac, alpha);

    // Shortscale → decays shorten with pitch
    float beta = 0.40f + 0.50f * x->shortscale_amt;
    v->decay_pitch_mul = powf(xfac, -beta);

    // Linger → velocity extends decays
    v->decay_vel_mul = (1.f + (0.30f + 1.20f * x->linger_amt) * jb_clamp(v->vel,0.f,1.f));

    // Brightness: purely user-controlled, no pitch/velocity dependence
    v->brightness_v = jb_clamp(x->brightness, 0.f, 1.f);

    // Bloom → bandwidth
    float baseBW = x->bandwidth;
    float addBW  = (0.15f + 0.45f * x->bloom_amt) * jb_clamp(v->vel,0.f,1.f);
    v->bandwidth_v = jb_clamp(baseBW + addBW, 0.f, 1.f);

    // per-mode dispersion targets (ignore fundamental)
    float total_disp = jb_clamp(x->dispersion + v->stiffen_add, 0.f, 1.f);
    if (x->dispersion_last<0.f){ x->dispersion_last = -1.f; }
    for(int i=0;i<x->n_modes;i++){
        if (!x->base[i].active || i==0){ v->disp_target[i]=0.f; continue; }
        float sig = x->base[i].disp_signature;
        v->disp_target[i] = jb_clamp(sig * total_disp, -1.f, 1.f);
    }
}
static void jb_update_crossring(t_juicy_bank_tilde *x, int self_idx){
    const float eps = 0.015f + 0.030f * x->crossring_amt;
    const float gmul = 1.f + (0.05f + 0.15f * x->crossring_amt);
    const float dmul = 1.f + (0.08f + 0.22f * x->crossring_amt);

    for(int m=0;m<JB_MAX_MODES;m++){
        x->v[self_idx].cr_gain_mul[m]=1.f;
        x->v[self_idx].cr_decay_mul[m]=1.f;
    }
    if (x->crossring_amt<=0.f) return;
    jb_voice_t *vs = &x->v[self_idx];
    if (vs->state==V_IDLE) return;

    for(int u=0; u<x->max_voices; ++u){
        if (u==self_idx) continue;
        const jb_voice_t *vu = &x->v[u];
        if (vu->state==V_IDLE) continue;

        for(int m=0;m<x->n_modes;m++){
            if (!x->base[m].active) continue;
            float rm = vs->m[m].ratio_now;
            float rel = (vu->f0>0.f) ? (rm * vs->f0 / vu->f0) : rm;
            if (jb_is_near_integer(rel, eps)){
                x->v[self_idx].cr_gain_mul[m]  *= gmul;
                x->v[self_idx].cr_decay_mul[m] *= dmul;
            }
        }
    }
}

// ---------- update voice coeffs ----------

// ---------- stretch (message only) + apply ----------
static void juicy_bank_tilde_stretch(t_juicy_bank_tilde *x, t_floatarg f){
    if (f < -1.f) f = -1.f;
    if (f >  1.f) f =  1.f;
    x->stretch = f;
}

static void juicy_bank_tilde_warp(t_juicy_bank_tilde *x, t_floatarg f){
    if (f < -1.f) f = -1.f;
    if (f >  1.f) f =  1.f;
    x->warp = f;
}
static void jb_apply_stretch(const t_juicy_bank_tilde *x, jb_voice_t *v){
    float k = 0.35f * jb_clamp(x->stretch, -1.f, 1.f);
    if (k == 0.f) { return; }
    float w = jb_clamp(x->warp, -1.f, 1.f);
    const float alpha = 4.0f; // curvature strength
    int denom = (x->n_modes - 1) > 0 ? (x->n_modes - 1) : 1;
    for (int i = 0; i < x->n_modes; ++i){
        if (i == 0) continue;                // keep fundamental ratio = 1x
        if (!x->base[i].keytrack) continue;  // absolute-Hz modes unaffected

        float r = v->m[i].ratio_now;
        if (r < 0.01f) r = 0.01f;

        float t = (float)i / (float)denom; // 0..1
        float bias;
        if (w >= 0.f){
            bias = powf(t, 1.f + alpha * w);
        } else {
            bias = 1.f - powf(1.f - t, 1.f + alpha * (-w));
        }

        float expo = 1.f + k * bias;
        if (expo < 0.1f) expo = 0.1f;
        if (expo > 3.0f) expo = 3.0f;
        v->m[i].ratio_now = powf(r, expo);
    }

}


static void jb_apply_offset(const t_juicy_bank_tilde *x, jb_voice_t *v){
    float amt = jb_clamp(x->offset_amt, -1.f, 1.f);
    if (amt == 0.f)
        return;
    float ratio = powf(2.f, amt); // offset in octaves
    for (int i = 0; i < x->n_modes; ++i){
        if (!x->base[i].active) continue;
        if (!x->base[i].keytrack) continue;
        if (i % 2 == 1){ // every other partial (skip fundamental at index 0)
            v->m[i].ratio_now *= ratio;
        }
    }
}


static void jb_update_voice_coeffs(t_juicy_bank_tilde *x, jb_voice_t *v){
    for(int i=0;i<x->n_modes;i++){ v->disp_offset[i] = v->disp_target[i]; }

    jb_apply_density(x, v);

    jb_lock_fundamental_after_density(x, v);
    jb_apply_stretch(x, v);
    jb_apply_offset(x, v);

    // --- pitch modulation via modulation matrix (currently: LFO1/LFO2 -> pitch) ---
    float f0_eff = v->f0;
    if (f0_eff <= 0.f) f0_eff = x->basef0_ref;
    if (f0_eff <= 0.f) f0_eff = 1.f;

    // pitch_mod is in -1..+1 roughly when only one source is active,
    // summed if multiple sources are used.
    float pitch_mod = 0.f;

    // source index 3 = lfo1, source index 4 = lfo2, target index 12 = pitch
    pitch_mod += x->lfo_val[0] * x->mod_matrix[3][12];
    pitch_mod += x->lfo_val[1] * x->mod_matrix[4][12];

    if (pitch_mod != 0.f){
        float semis = pitch_mod * JB_PITCH_MOD_SEMITONES; // +/- range in semitones
        float ratio = powf(2.f, semis / 12.f);
        f0_eff *= ratio;
    }

    float md_amt = jb_clamp(x->micro_detune,0.f,1.f);
    float bw_amt = jb_clamp(v->bandwidth_v, 0.f, 1.f);

    for(int i=0;i<x->n_modes;i++){
        jb_mode_rt_t *md=&v->m[i];
        if(!x->base[i].active){
            md->a1L=md->a2L=md->a1bL=md->a2bL=0.f;
            md->a1R=md->a2R=md->a1bR=md->a2bR=0.f;
            md->t60_s=0.f;
            md->normL = md->normR = 1.f;
            continue;
        }

        // base ratio including dispersion
        float ratio_base = md->ratio_now + v->disp_offset[i];
        if (x->dispersion < 0.f){ float a = jb_clamp(-x->dispersion, 0.f, 1.f); float nearest = roundf(ratio_base); ratio_base = (1.f - a)*ratio_base + a*nearest; }

        // density pivot bias now fixed to neighbor mapping (per-resonator); no harmonic stepping here.
// micro detune per-ear (except fundamental)
        float ratioL = ratio_base;
        float ratioR = ratio_base;
        if(i!=0){ ratioL += md_amt * md->md_hit_offsetL; ratioR += md_amt * md->md_hit_offsetR; }
        if (ratioL < 0.01f) ratioL = 0.01f;
        if (ratioR < 0.01f) ratioR = 0.01f;

        float HzL = x->base[i].keytrack ? (f0_eff * ratioL) : ratioL;
        float HzR = x->base[i].keytrack ? (f0_eff * ratioR) : ratioR;
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
            md->a1L=md->a2L=md->a1bL=md->a2bL=0.f;
            md->a1R=md->a2R=md->a1bR=md->a2bR=0.f;
            md->normL = md->normR = 1.f;
        }float base_ms = x->base[i].base_decay_ms;
        // T60 & radius
        float T60 = jb_clamp(base_ms, 0.f, 1e7f) * 0.001f;
        T60 *= v->decay_pitch_mul;
        T60 *= v->decay_vel_mul;
        T60 *= v->cr_decay_mul[i];
        {
            
            /* Per-mode damping focus: weight along modes with wrap */
            float b = jb_clamp(x->damp_broad, 0.f, 1.f);
            float p = x->damp_point;
            if (p < 0.f) p = 0.f;
            if (p > 1.f) p = 1.f;
            float k_norm = (x->n_modes>1)? ((float)i/(float)(x->n_modes-1)) : 0.f;
            float dx = fabsf(k_norm - p); if (dx > 0.5f) dx = 1.f - dx; /* circular distance */
            float n = (float)((x->n_modes>0)?x->n_modes:1);
            float sigma_min = 0.5f / n;            /* ~single-mode width */
            float sigma_max = 0.5f;                /* whole bank */
            float sigma = (1.f - b)*sigma_max + b*sigma_min;
            float wloc = expf(-0.5f * (dx*dx) / (sigma*sigma)); /* 0..1 */
            /*/* apply local weighting to global damping amount */
            float d_amt = jb_clamp(x->damping, -1.f, 1.f) * wloc;
            if (d_amt >= 0.f){
                T60 *= (1.f - d_amt);
            } else {
                float Dneg = -d_amt;
                float ceiling = JB_MAX_DECAY_S;
                T60 = T60 + Dneg * (ceiling - T60);
                if (T60 > ceiling) T60 = ceiling;
            }

        }
        md->t60_s = T60;

        float r = (T60 <= 0.f) ? 0.f : powf(10.f, -3.f / (T60 * x->sr));
        float cL=cosf(wL), cR=cosf(wR);

        md->a1L=2.f*r*cL; md->a2L=-r*r;
        md->a1R=2.f*r*cR; md->a2R=-r*r;

        // --- NEW: frequency-normalized resonance drive factors ---
        float denomL = (1.f - 2.f*r*cL + r*r);
        float denomR = (1.f - 2.f*r*cR + r*r);
        if (denomL < 1e-6f) denomL = 1e-6f;
        if (denomR < 1e-6f) denomR = 1e-6f;
        md->normL = denomL;
        md->normR = denomR;

        if (bw_amt > 0.f){
            float mode_scale = (x->n_modes>1)? ((float)i/(float)(x->n_modes-1)) : 0.f;
            float max_det = 0.0005f + 0.0015f * mode_scale;
            float detL = jb_clamp(md->bw_hit_ratioL, -max_det, max_det) * bw_amt;
            float detR = jb_clamp(md->bw_hit_ratioR, -max_det, max_det) * bw_amt;
            float w2L = wL * (1.f + detL);
            float w2R = wR * (1.f + detR);
            float c2L = cosf(w2L);
            float c2R = cosf(w2R);
            md->a1bL = 2.f*r*c2L; md->a2bL = -r*r;
            md->a1bR = 2.f*r*c2R; md->a2bR = -r*r;
        } else {
            md->a1bL=md->a2bL=0.f; md->a1bR=md->a2bR=0.f;
        }
    }
}

// ---------- update voice gains ----------
static void jb_update_voice_gains(const t_juicy_bank_tilde *x, jb_voice_t *v){
    for(int i=0;i<x->n_modes;i++){
        if(!x->base[i].active){ v->m[i].gain_now=0.f; continue; }

        float ratio = v->m[i].ratio_now + v->disp_offset[i];
        // ratio_rel here only for weighting functions (brightness, anisotropy, etc.):
        float ratio_rel = x->base[i].keytrack ? ratio : ((v->f0>0.f)? (ratio / v->f0) : ratio);

        float g = x->base[i].base_gain * jb_bright_gain(ratio_rel, v->brightness_v);

        float w = 1.f;
        if (i != 0 && x->aniso != 0.f){
            float r = ratio_rel;
            float kf = nearbyintf(r);
            int   k  = (int)kf; if (k < 1) k = 1;
            float d  = fabsf(r - kf);
            float w0 = 0.25f;              /* fixed internal width */
            float prox = expf(- (d/w0)*(d/w0)); /* 0..1 */
            int parity = (k % 2 == 0) ? +1 : -1; /* +1 even, -1 odd */
            float bias = x->aniso * parity * prox;
            float ampMul = 1.f + bias;
            if (ampMul < 0.f) ampMul = 0.f;
            w *= ampMul;
        }

        float wp = jb_position_weight(ratio_rel, x->position);

        g *= v->cr_gain_mul[i];

        
float gn = g * w * wp;
        if (v->m[i].nyq_kill) gn = 0.f;
            if (x->active_modes < x->n_modes) { if (i >= x->active_modes) gn = 0.f; }
        // --- SINE AM mask (bipolar focus vs complement, applied after gn is computed) ---
        {
            int N = x->n_modes;
            float pitch = jb_clamp(x->sine_pitch, 0.f, 1.f);
            float depth = jb_clamp(x->sine_depth, -1.f, 1.f);
            float phase = x->sine_phase;

            // build a smooth 0..1 pattern mask along modes
            float cycles_min = 0.25f;
            float cycles_max = floorf((float)N * 0.5f);
            if (cycles_max < cycles_min) cycles_max = cycles_min;
            float cycles = cycles_min + pitch * (cycles_max - cycles_min);
            float k_norm = (N>1) ? ((float)i / (float)(N-1)) : 0.f;
            float theta = 2.f * (float)M_PI * (cycles * k_norm + phase);
            float w01 = 0.5f * (1.f + cosf(theta)); // 1 at pattern center, 0 at pattern "nulls"

            // make the pattern "sharper" as |depth| increases
            float sharp = 1.0f + 8.0f * fabsf(depth);
            float pattern = powf(w01, sharp); // pattern membership mask in [0,1]

            // bipolar amount:
            //   depth > 0: attenuate pattern partials
            //   depth < 0: attenuate non-pattern partials
            float a_pos = (depth > 0.f) ? depth : 0.f;
            float a_neg = (depth < 0.f) ? -depth : 0.f;

            float weight = 1.f
                           - a_pos * pattern          // turn down pattern when depth > 0
                           - a_neg * (1.f - pattern); // turn down complement when depth < 0

            if (weight < 0.f) weight = 0.f;
            gn *= weight;
        }
        v->m[i].gain_now = (gn<0.f)?0.f:gn;
    }
}

// ---------- allocator helpers ----------
static void jb_voice_reset_states(const t_juicy_bank_tilde *x, jb_voice_t *v, jb_rng_t *rng){
        v->rel_env = 1.f;
v->energy = 0.f;
    for(int i=0;i<x->n_modes;i++){
        jb_mode_rt_t *md=&v->m[i];
        md->ratio_now = x->base[i].base_ratio;
        md->decay_ms_now = x->base[i].base_decay_ms;
        md->gain_now = x->base[i].base_gain;
        md->t60_s = md->decay_ms_now*0.001f; md->decay_u=0.f;
        md->a1L=md->a2L=md->y1L=md->y2L=0.f; md->a1bL=md->a2bL=md->y1bL=md->y2bL=0.f;
        md->a1R=md->a2R=md->y1R=md->y2R=0.f; md->a1bR=md->a2bR=md->y1bR=md->y2bR=0.f;
        md->driveL=md->driveR=0.f; md->envL=md->envR=0.f;
        md->y_pre_lastL=md->y_pre_lastR=0.f;
        md->hit_gateL=md->hit_gateR=0; md->hit_coolL=md->hit_coolR=0;
        md->md_hit_offsetL = 0.f; md->md_hit_offsetR = 0.f;
        md->bw_hit_ratioL = 0.f;  md->bw_hit_ratioR = 0.f;
        md->normL = md->normR = 1.f;
        md->nyq_kill = 0;
        v->disp_offset[i]=0.f; v->disp_target[i]=0.f;
        v->cr_gain_mul[i]=1.f; v->cr_decay_mul[i]=1.f;
        (void)rng;
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

static void jb_note_on(t_juicy_bank_tilde *x, float f0, float vel){
    int idx = jb_find_voice_to_steal(x);
    jb_voice_t *v = &x->v[idx];
    v->state = V_HELD; v->f0 = (f0<=0.f)?1.f:f0; v->vel = jb_clamp(vel,0.f,1.f);
    jb_voice_reset_states(x, v, &x->rng);
    jb_project_behavior_into_voice(x, v);
}

static void jb_note_off(t_juicy_bank_tilde *x, float f0){
    int match=-1; float best=1e9f; float tol=0.5f;
    for(int i=0;i<x->max_voices;i++){
        if (x->v[i].state==V_HELD){
            float d=fabsf(x->v[i].f0 - f0);
            if (d<tol){ match=i; break; }
            if (x->v[i].energy<best){ best=x->v[i].energy; match=i; }
        }
    }
    if (match>=0) x->v[match].state = V_RELEASE;
}

// ===== Explicit voice-addressed control (for Pd [poly]) =====
static void jb_note_on_voice(t_juicy_bank_tilde *x, int vix1, float f0, float vel){
    if (vix1 < 1) vix1 = 1;
    if (vix1 > x->max_voices) vix1 = x->max_voices;
    int idx = vix1 - 1;
    if (f0 <= 0.f) f0 = 1.f;
    if (vel < 0.f) vel = 0.f;
    if (vel > 1.f) vel = 1.f;
    jb_voice_t *v = &x->v[idx];
    v->state = V_HELD; v->f0 = f0; v->vel = vel;
    jb_voice_reset_states(x, v, &x->rng);
    jb_project_behavior_into_voice(x, v);
}

static void jb_note_off_voice(t_juicy_bank_tilde *x, int vix1){
    if (vix1 < 1) vix1 = 1;
    if (vix1 > x->max_voices) vix1 = x->max_voices;
    int idx = vix1 - 1;
    if (x->v[idx].state != V_IDLE) x->v[idx].state = V_RELEASE;
}

// Message handlers (voice-addressed)
static void juicy_bank_tilde_note_poly(t_juicy_bank_tilde *x, t_floatarg vix, t_floatarg f0, t_floatarg vel){
    if (vel <= 0.f) { jb_note_off_voice(x, (int)vix); }
    else            { jb_note_on_voice(x, (int)vix, f0, vel); }
}

static void juicy_bank_tilde_note_poly_midi(t_juicy_bank_tilde *x, t_floatarg vix, t_floatarg midi, t_floatarg vel){
    if (vel <= 0.f) { jb_note_off_voice(x, (int)vix); }
    else            { jb_note_on_voice(x, (int)vix, jb_midi_to_hz(midi), vel); }
}

static void juicy_bank_tilde_off_poly(t_juicy_bank_tilde *x, t_floatarg vix){
    jb_note_off_voice(x, (int)vix);
}

// ---------- perform ----------

static t_int *juicy_bank_tilde_perform(t_int *w){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)(w[1]);
    // inputs
    t_sample *inL=(t_sample *)(w[2]);
    t_sample *inR=(t_sample *)(w[3]);
    t_sample *v1L=(t_sample *)(w[4]);
    t_sample *v1R=(t_sample *)(w[5]);
    t_sample *v2L=(t_sample *)(w[6]);
    t_sample *v2R=(t_sample *)(w[7]);
    t_sample *v3L=(t_sample *)(w[8]);
    t_sample *v3R=(t_sample *)(w[9]);
    t_sample *v4L=(t_sample *)(w[10]);
    t_sample *v4R=(t_sample *)(w[11]);
    // outputs
    t_sample *outL=(t_sample *)(w[12]);
    t_sample *outR=(t_sample *)(w[13]);
    int n=(int)(w[14]);

    // clear outputs
    for(int i=0;i<n;i++){ outL[i]=0; outR[i]=0; }

    // update LFOs once per block (for modulation matrix sources)
    jb_update_lfos_block(x, n);

    // Per-block updates that don't change sample-phase
    for(int vix=0; vix<x->max_voices; ++vix){
        jb_voice_t *v = &x->v[vix];
        if (v->state==V_IDLE) continue;
        jb_update_crossring(x, vix);
        jb_project_behavior_into_voice(x, v); // keep behavior up-to-date
        jb_update_voice_coeffs(x, v);
        jb_update_voice_gains(x, v);
    }


    t_sample *vinL[JB_MAX_VOICES] = { v1L, v2L, v3L, v4L };
    t_sample *vinR[JB_MAX_VOICES] = { v1R, v2R, v3R, v4R };

    // constants
    const float aHP = x->fb_hp_a;
    float fbz = x->fb_amt_z;
            
    const float fba = x->fb_slew_a;

    // Process per-voice, sample-major so feedback uses only a 2-sample delay (no block latency)
    for(int vix=0; vix<x->max_voices; ++vix){
        jb_voice_t *v = &x->v[vix];
        if (v->state==V_IDLE) continue;

        const float bw_amt = jb_clamp(v->bandwidth_v, 0.f, 1.f);
        const float use_gate = (x->exciter_mode==0) ? ((v->state==V_HELD)?1.f:0.f) : 1.f;
        t_sample *srcL = (x->exciter_mode==0) ? inL : vinL[vix];
        t_sample *srcR = (x->exciter_mode==0) ? inR : vinR[vix];
        // global modal bank master gain (0..1) from former fb_drive inlet,
        // modulated per-voice by the modulation matrix target "master" (index 11).
        float fb_base = jb_clamp(x->fb_drive, 0.f, 1.f);

        // master_mod accumulates contributions from all sources mapped to "master".
        // For now we only use LFO1 (source 3) and LFO2 (source 4).
        float master_mod = 0.f;
        master_mod += x->lfo_val[0] * x->mod_matrix[3][11]; // lfo1_to_master
        master_mod += x->lfo_val[1] * x->mod_matrix[4][11]; // lfo2_to_master

        // Clamp summed modulation to [-1,1] so it stays well-behaved even if multiple sources are active.
        if (master_mod > 1.f) master_mod = 1.f;
        else if (master_mod < -1.f) master_mod = -1.f;

        // Map modulation into the 0..1 range around fb_base without overshooting:
        //   master_mod > 0  -> move towards 1
        //   master_mod < 0  -> move towards 0
        float bank_gain;
        if (master_mod >= 0.f){
            bank_gain = fb_base + master_mod * (1.f - fb_base);
        } else {
            bank_gain = fb_base + master_mod * fb_base;
        }

        // feedback filter & delay states per-voice/per-ear
        float hp_x1L = v->fb_hp_x1L, hp_y1L = v->fb_hp_y1L;
        float hp_x1R = v->fb_hp_x1R, hp_y1R = v->fb_hp_y1R;
        float d1L = v->fb_d1L, d2L = v->fb_d2L;
        float d1R = v->fb_d1R, d2R = v->fb_d2R;
        float lpL  = v->fb_lpL,  lpR  = v->fb_lpR;
        float pL   = v->fb_pulseL, pR = v->fb_pulseR;

        // per-voice feedback shaping constants (depend on pitch and global params)
        float fb_timer = jb_clamp(x->fb_timer, 0.f, 1.f);
        float fb_fmax_norm = jb_clamp(x->fb_fmax, 0.f, 1.f);
        float f0 = v->f0;
        if (f0 <= 0.f) f0 = x->basef0_ref;
        float fmax_ratio = powf(2.f, 2.f * fb_fmax_norm); // up to +2 octaves
        float fmax = f0 * fmax_ratio;
        float nyq = 0.5f * x->sr;
        if (fmax < 10.f) fmax = 10.f;
        if (fmax > nyq) fmax = nyq;
        float k = 2.f * (float)M_PI * fmax / x->sr;
        if (k > 1.f) k = 1.f;
        float a_lp = expf(-k);
        float b_lp = 1.f - a_lp;

        for(int i=0;i<n;i++){
            // Slew fb amount per-sample (shared param)
            float fb_tgt = 0.99f * jb_clamp(x->fb_amt, -1.f, 1.f);
            fbz = fba * fbz + (1.f - fba) * fb_tgt;

            // compute feedback injection for this sample from 2-sample delayed, shaped voice output
            float loop_g = fbz;
            float fbL = loop_g * d2L;
            float fbR = loop_g * d2R;
float vsumL = 0.f, vsumR = 0.f; // this sample's voice sum

            // Per-mode one-sample step
            for(int m=0;m<x->n_modes;m++){
                if(!x->base[m].active) continue;
                jb_mode_rt_t *md=&v->m[m];
                if (md->gain_now<=0.f || md->nyq_kill) continue;

                // pull local copies
                float y1L=md->y1L, y2L=md->y2L, y1bL=md->y1bL, y2bL=md->y2bL, driveL=md->driveL, envL=md->envL;
                float y1R=md->y1R, y2R=md->y2R, y1bR=md->y1bR, y2bR=md->y2bR, driveR=md->driveR, envR=md->envR;
                float u = md->decay_u;
                float att_ms = jb_clamp(x->base[m].attack_ms,0.f,500.f);
                float att_a = (att_ms<=0.f)?1.f:(1.f-expf(-1.f/(0.001f*att_ms*x->sr)));
                float du = (md->t60_s > 1e-6f) ? (1.f / (md->t60_s * x->sr)) : 1.f;

                // excitation: use current sample src + feedback for this sample
                float excL = use_gate * (srcL[i] + fbL) * md->gain_now;
                float excR = use_gate * (srcR[i] + fbR) * md->gain_now;

                // modal integrators
                driveL += att_a*(excL - driveL);
                float y_linL = (md->a1L*y1L + md->a2L*y2L) + driveL * md->normL;
                y2L=y1L; y1L=y_linL;

                driveR += att_a*(excR - driveR);
                float y_linR = (md->a1R*y1R + md->a2R*y2R) + driveR * md->normR;
                y2R=y1R; y1R=y_linR;

                float y_totalL = y_linL;
                float y_totalR = y_linR;
                if (bw_amt > 0.f){
                    float y_lin_bL = (md->a1bL*y1bL + md->a2bL*y2bL);
                    y2bL=y1bL; y1bL=y_lin_bL;
                    y_totalL += 0.12f * bw_amt * y_lin_bL;

                    float y_lin_bR = (md->a1bR*y1bR + md->a2bR*y2bR);
                    y2bR=y1bR; y1bR=y_lin_bR;
                    y_totalR += 0.12f * bw_amt * y_lin_bR;
                }

                float S = jb_curve_shape_gain(u, x->base[m].curve_amt);
                if (x->base[m].curve_amt < 0.f){ if (S < 0.001f) S = 0.001f; }
                y_totalL *= S; y_totalR *= S;
                u += du; if(u>1.f){ u=1.f; }

                // equal-power pan and sum (scaled by global bank_gain)
                float p = jb_clamp(x->base[m].pan, -1.f, 1.f);
                float wL = sqrtf(0.5f*(1.f - p));
                float wR = sqrtf(0.5f*(1.f + p));
                y_totalL *= v->rel_env; y_totalR *= v->rel_env;
                float voiceL = y_totalL * wL * bank_gain;
                float voiceR = y_totalR * wR * bank_gain;
                outL[i] += voiceL;
                outR[i] += voiceR;
                vsumL += voiceL;
                vsumR += voiceR;

                // write back small subset (leave other slow vars unchanged)
                md->y1L=y1L; md->y2L=y2L; md->y1bL=y1bL; md->y2bL=y2bL; md->driveL=driveL; md->envL=envL;
                md->y1R=y1R; md->y2R=y2R; md->y1bR=y1bR; md->y2bR=y2bR; md->driveR=driveR; md->envR=envR;
                md->decay_u=u;
                md->y_pre_lastL = y_totalL; md->y_pre_lastR = y_totalR;
            } // end modes

            // Update feedback filter/delay from this sample's voice sum so next sample sees it
            // 30 Hz HP (leaky differentiator)
                        // 30 Hz HP (leaky differentiator)
            float hl = aHP * (hp_y1L + vsumL - hp_x1L); hp_x1L = vsumL; hp_y1L = hl;
            float hr = aHP * (hp_y1R + vsumR - hp_x1R); hp_x1R = vsumR; hp_y1R = hr;

            // Feedback limiter: hard clip to +/-0.99 before further shaping
            const float fb_lim = 0.99f;
            float rawL = hl;
            if (rawL > fb_lim)  rawL = fb_lim;
            else if (rawL < -fb_lim) rawL = -fb_lim;
            float rawR = hr;
            if (rawR > fb_lim)  rawR = fb_lim;
            else if (rawR < -fb_lim) rawR = -fb_lim;

            // One-pole lowpass in the feedback path (cutoff controlled by fb_fmax)
            lpL = a_lp * lpL + b_lp * rawL;
            lpR = a_lp * lpR + b_lp * rawR;

            // Derive pulse signal via comparator + smoothing
            float sqL = (lpL >= 0.f) ? 1.f : -1.f;
            float sqR = (lpR >= 0.f) ? 1.f : -1.f;
            pL = a_lp * pL + b_lp * sqL;
            pR = a_lp * pR + b_lp * sqR;

            // Crossfade between smoothed continuous feedback and pulse-shaped version
            float shapedL = (1.f - fb_timer) * lpL + fb_timer * pL;
            float shapedR = (1.f - fb_timer) * lpR + fb_timer * pR;

            // 2-sample delay in feedback loop
            d2L = d1L; d1L = shapedL;
            d2R = d1R; d1R = shapedR;

            // per-sample release envelope update (decays in V_RELEASE, 20ms..5s)
            if (v->state == V_RELEASE){
                float tau = 0.02f + 4.98f * jb_clamp(x->release_amt, 0.f, 1.f);
                float a_rel = expf(-1.0f / (x->sr * tau));
                v->rel_env *= a_rel;
                if (v->rel_env < 1e-5f) {
                    v->rel_env = 0.f;
                    v->state   = V_IDLE;
                }
            } else if (v->state == V_HELD) {
                // fully open envelope while key is held
                v->rel_env = 1.f;
            } else {
                // idle voices stay at 0 to avoid re-opening residual ring
                v->rel_env = 0.f;
            }

        } // end samples

        // write back states
        v->fb_hp_x1L = hp_x1L; v->fb_hp_y1L = hp_y1L;
        v->fb_hp_x1R = hp_x1R; v->fb_hp_y1R = hp_y1R;
        v->fb_d1L = d1L; v->fb_d2L = d2L;
        v->fb_d1R = d1R; v->fb_d2R = d2R;
        v->fb_lpL = lpL;   v->fb_lpR = lpR;
        v->fb_pulseL = pL; v->fb_pulseR = pR;
    } // end voices

    x->fb_amt_z = fbz;

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

    return (w + 15);
}


// ---------- base setters & messages ----------
static void juicy_bank_tilde_modes(t_juicy_bank_tilde *x, t_floatarg nf){
    int n=(int)nf; if(n<1)n=1; if(n>JB_MAX_MODES)n=JB_MAX_MODES; x->n_modes=n;
    x->active_modes = x->n_modes;
    if (x->edit_idx >= x->n_modes) x->edit_idx = x->n_modes-1;
    for(int i=0;i<x->n_modes;i++){
        if(x->base[i].base_ratio<=0.f) x->base[i].base_ratio=(float)(i+1);
    }
}
static void juicy_bank_tilde_active(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg onf){
    int idx=(int)idxf-1; if(idx<0||idx>=x->n_modes) return; x->base[idx].active=(onf>0.f)?1:0;
}

// INDIVIDUAL (per-mode via index)
static void juicy_bank_tilde_index(t_juicy_bank_tilde *x, t_floatarg f){
    int idx=(int)f; if(idx<1) idx=1; if(idx>x->n_modes) idx=x->n_modes; x->edit_idx=idx-1;
}
static void juicy_bank_tilde_ratio_i(t_juicy_bank_tilde *x, t_floatarg r){
    int i=x->edit_idx; if(i<0||i>=x->n_modes) return;
    if (x->base[i].keytrack){
        float v = (r<=0.f)?0.01f:r;
        x->base[i].base_ratio = v;
    } else {
        float ui = r; if (ui < 0.f) ui = 0.f; if (ui > 10.f) ui = 10.f;
        x->base[i].base_ratio = 100.f * ui;
    }
}
static void juicy_bank_tilde_gain_i(t_juicy_bank_tilde *x, t_floatarg g){
    int i=x->edit_idx; if(i<0||i>=x->n_modes) return; x->base[i].base_gain=jb_clamp(g,0.f,1.f);
}
static void juicy_bank_tilde_attack_i(t_juicy_bank_tilde *x, t_floatarg ms){
    int i=x->edit_idx; if(i<0||i>=x->n_modes) return; x->base[i].attack_ms=(ms<0.f)?0.f:ms;
}
static void juicy_bank_tilde_decay_i(t_juicy_bank_tilde *x, t_floatarg ms){
    int i=x->edit_idx; if(i<0||i>=x->n_modes) return; x->base[i].base_decay_ms=(ms<0.f)?0.f:ms;
}
static void juicy_bank_tilde_curve_i(t_juicy_bank_tilde *x, t_floatarg amt){
    int i=x->edit_idx; if(i<0||i>=x->n_modes) return; if(amt<-1.f)amt=-1.f; if(amt>1.f)amt=1.f; x->base[i].curve_amt=amt;
}
static void juicy_bank_tilde_pan_i(t_juicy_bank_tilde *x, t_floatarg p){
    int i=x->edit_idx; if(i<0||i>=x->n_modes) return; x->base[i].pan=jb_clamp(p,-1.f,1.f);
}
static void juicy_bank_tilde_keytrack_i(t_juicy_bank_tilde *x, t_floatarg kt){
    int i=x->edit_idx; if(i<0||i>=x->n_modes) return; x->base[i].keytrack = (kt>0.f)?1:0;
}

// Per-mode lists
static void juicy_bank_tilde_freq(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s; for(int i=0;i<argc && i<JB_MAX_MODES;i++){ if(argv[i].a_type==A_FLOAT){ float v=atom_getfloat(argv+i); x->base[i].base_ratio=(v<=0.f)?0.01f:v; } }
}
static void juicy_bank_tilde_decays(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s; for(int i=0;i<argc && i<JB_MAX_MODES;i++){ if(argv[i].a_type==A_FLOAT){ float v=atom_getfloat(argv+i); x->base[i].base_decay_ms=(v<0.f)?0.f:v; } }
}
static void juicy_bank_tilde_amps(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s; for(int i=0;i<argc && i<JB_MAX_MODES;i++){ if(argv[i].a_type==A_FLOAT){ float v=atom_getfloat(argv+i); x->base[i].base_gain=jb_clamp(v,0.f,1.f); } }
}

// BODY globals
static void juicy_bank_tilde_damp_broad(t_juicy_bank_tilde *x, t_floatarg f){ x->damp_broad=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_damp_point(t_juicy_bank_tilde *x, t_floatarg f){ x->damp_point=jb_wrap01(f); }
static void juicy_bank_tilde_damping(t_juicy_bank_tilde *x, t_floatarg f){ x->damping=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){ x->position=(f<=0.f)?0.f:jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_density(t_juicy_bank_tilde *x, t_floatarg f){
    // Interpret density as "harmonic gap units":
    //   0 -> 1x spacing, 1 -> 2x spacing, 2 -> 3x spacing, ...
    // Negative side is limited to -1 (one whole-number gap less, i.e. gap cannot go below 0).
    float v = (float)f;
    if (v < -1.f)
        v = -1.f;
    x->density_amt = v;
}

static void juicy_bank_tilde_density_pivot(t_juicy_bank_tilde *x){ x->density_mode=DENSITY_PIVOT; }
static void juicy_bank_tilde_density_individual(t_juicy_bank_tilde *x){ x->density_mode=DENSITY_INDIV; }
static void juicy_bank_tilde_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_aniso_eps(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso_eps=jb_clamp(f,0.f,0.25f); }
static void juicy_bank_tilde_release(t_juicy_bank_tilde *x, t_floatarg f){ x->release_amt = jb_clamp(f, 0.f, 1.f); }

// realism & misc
static void juicy_bank_tilde_phase_random(t_juicy_bank_tilde *x, t_floatarg f){ x->phase_rand=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_phase_debug(t_juicy_bank_tilde *x, t_floatarg on){ x->phase_debug=(on>0.f)?1:0; }
static void juicy_bank_tilde_bandwidth(t_juicy_bank_tilde *x, t_floatarg f){ x->bandwidth=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_micro_detune(t_juicy_bank_tilde *x, t_floatarg f){ x->micro_detune=jb_clamp(f,0.f,1.f); }
// --- SINE param setters ---
static void juicy_bank_tilde_sine_pitch(t_juicy_bank_tilde *x, t_floatarg f){
    x->sine_pitch = jb_clamp(f, 0.f, 1.f);
}
static void juicy_bank_tilde_sine_depth(t_juicy_bank_tilde *x, t_floatarg f){
    x->sine_depth = jb_clamp(f, -1.f, 1.f); }
static void juicy_bank_tilde_sine_phase(t_juicy_bank_tilde *x, t_floatarg f){
    float p = f - floorf(f);
    if (p < 0.f) p += 1.f;
    x->sine_phase = p;
}

// --- LFO + ADSR param setters (for modulation matrix) ---
static void juicy_bank_tilde_lfo_shape(t_juicy_bank_tilde *x, t_floatarg f){
    int s = (int)floorf(f + 0.5f);
    if (s < 1) s = 1;
    if (s > 4) s = 4;
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
}
static void juicy_bank_tilde_adsr_ms(t_juicy_bank_tilde *x, t_floatarg f){
    // treat this as a real-time ADSR envelope value in 0..1
    float v = jb_clamp(f, 0.f, 1.f);
    x->adsr_ms = v;
}

static void juicy_bank_tilde_offset(t_juicy_bank_tilde *x, t_floatarg f){
    x->offset_amt = jb_clamp(f, -1.f, 1.f);
}

// dispersion & seeds

static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){
    float v = jb_clamp(f, -1.f, 1.f);
    float pos = (v>0.f)?v:0.f;
    if (x->dispersion_last < 0.f || fabsf(pos - x->dispersion_last) > 1e-6f){
        for(int i=0;i<x->n_modes;i++){ x->base[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng); }
        x->dispersion_last = pos;
    }
    x->dispersion = v;
}
static void juicy_bank_tilde_seed(t_juicy_bank_tilde *x, t_floatarg f){
    jb_rng_seed(&x->rng, (unsigned int)((int)f*2654435761u));
    for(int i=0;i<x->n_modes;i++){
        x->base[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng);
        x->base[i].micro_sig      =(i==0)?0.f:jb_rng_bi(&x->rng);
    }
    x->dispersion_last=x->dispersion;
}
static void juicy_bank_tilde_dispersion_reroll(t_juicy_bank_tilde *x){
    for(int i=0;i<x->n_modes;i++){ x->base[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng); }
    x->dispersion_last=-1.f;
    juicy_bank_tilde_dispersion(x, x->dispersion);
}

// BEHAVIOR amounts
static void juicy_bank_tilde_stiffen(t_juicy_bank_tilde *x, t_floatarg f){ x->stiffen_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_shortscale(t_juicy_bank_tilde *x, t_floatarg f){ x->shortscale_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_linger(t_juicy_bank_tilde *x, t_floatarg f){ x->linger_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_tilt(t_juicy_bank_tilde *x, t_floatarg f){ x->tilt_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_bite(t_juicy_bank_tilde *x, t_floatarg f){ x->bite_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_bloom(t_juicy_bank_tilde *x, t_floatarg f){ x->bloom_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_crossring(t_juicy_bank_tilde *x, t_floatarg f){ x->crossring_amt=jb_clamp(f,0.f,1.f); }

// Notes/poly (non-voice-addressed)
static void juicy_bank_tilde_note(t_juicy_bank_tilde *x, t_floatarg f0, t_floatarg vel){
    if (f0<=0.f){ f0=1.f; }
    jb_note_on(x, f0, vel);
}
static void juicy_bank_tilde_off(t_juicy_bank_tilde *x, t_floatarg f0){ jb_note_off(x, (f0<=0.f)?1.f:f0); }
static void juicy_bank_tilde_voices(t_juicy_bank_tilde *x, t_floatarg nf){
    (void)nf; x->max_voices = JB_MAX_VOICES; // fixed 4
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

// exciter mode toggle
static void juicy_bank_tilde_exciter_mode(t_juicy_bank_tilde *x, t_floatarg on){
    x->exciter_mode = (on>0.f)?1:0;
    post("juicy_bank~: exciter_mode = %d (%s)", x->exciter_mode, x->exciter_mode ? "per-voice 8-in" : "global 2-in");
}

// reset/restart
static void juicy_bank_tilde_reset(t_juicy_bank_tilde *x){
    for(int v=0; v<JB_MAX_VOICES; ++v){
        x->v[v].state = V_IDLE; x->v[v].f0 = x->basef0_ref; x->v[v].vel = 0.f; x->v[v].energy=0.f; x->v[v].rel_env = 1.f;

        // FEEDBACK per-voice init
        x->v[v].fb_hp_x1L = x->v[v].fb_hp_y1L = 0.f;
        x->v[v].fb_hp_x1R = x->v[v].fb_hp_y1R = 0.f;
        x->v[v].fb_d1L = x->v[v].fb_d2L = 0.f;
        x->v[v].fb_d1R = x->v[v].fb_d2R = 0.f;
        x->v[v].fb_lpL = x->v[v].fb_lpR = 0.f;
        x->v[v].fb_pulseL = x->v[v].fb_pulseR = 0.f;
        x->v[v].fb_prev_len = 0;
        for (int _i=0; _i<JB_FB_MAX; ++_i){ x->v[v].fb_prevL[_i]=0.f; x->v[v].fb_prevR[_i]=0.f; }
        for(int i=0;i<JB_MAX_MODES;i++){
            x->v[v].disp_offset[i]=x->v[v].disp_target[i]=0.f;
            x->v[v].cr_gain_mul[i]=x->v[v].cr_decay_mul[i]=1.f;
        }
    }
}
static void juicy_bank_tilde_restart(t_juicy_bank_tilde *x){ juicy_bank_tilde_reset(x); }

// ---------- dsp setup/free ----------
static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;

    // FEEDBACK coeffs
    x->fb_hp_a  = expf(-2.f * (float)M_PI * 30.f / x->sr);
    x->fb_slew_a = expf(-1.f / (0.010f * x->sr)); // ~10 ms slew
    float fc=8.f; float RC=1.f/(2.f*M_PI*fc); float dt=1.f/x->sr; x->hp_a=RC/(RC+dt);

    // sp layout: [inL, inR, v1L, v1R, v2L, v2R, v3L, v3R, v4L, v4R, outL, outR]
    t_int argv[2 + 12 + 1];
    int a=0;
    argv[a++] = (t_int)x;
    for(int k=0;k<12;k++) argv[a++] = (t_int)(sp[k]->s_vec);
    argv[a++] = (int)(sp[0]->s_n);
    dsp_addv(juicy_bank_tilde_perform, a, argv);
}

static void juicy_bank_tilde_free(t_juicy_bank_tilde *x){
    inlet_free(x->in_crossring);
    inlet_free(x->in_release);

    inlet_free(x->in_damping); inlet_free(x->in_damp_broad); inlet_free(x->in_damp_point); inlet_free(x->in_brightness); inlet_free(x->in_position);
    inlet_free(x->in_density);
    inlet_free(x->in_warp); inlet_free(x->in_dispersion);
    inlet_free(x->in_offset);
    inlet_free(x->in_aniso);

            inlet_free(x->in_stretch);
inlet_free(x->in_sine_pitch);
    inlet_free(x->in_sine_depth);
    inlet_free(x->in_sine_phase);
    inlet_free(x->in_fb_amt);
    inlet_free(x->in_fb_drive);
    inlet_free(x->in_fb_timer);
    inlet_free(x->in_fb_fmax);
    inlet_free(x->in_partials); // free 'partials' inlet
inlet_free(x->in_index); inlet_free(x->in_ratio); inlet_free(x->in_gain);
    inlet_free(x->in_attack); inlet_free(x->in_decay); inlet_free(x->in_curve); inlet_free(x->in_pan); inlet_free(x->in_keytrack);

    inlet_free(x->inR);
    for(int i=0;i<JB_MAX_VOICES;i++){ if(x->in_vL[i]) inlet_free(x->in_vL[i]); if(x->in_vR[i]) inlet_free(x->in_vR[i]); }

    outlet_free(x->outL); outlet_free(x->outR);

// ---------- FEEDBACK setters ----------
}

static void juicy_bank_tilde_fb_amt(t_juicy_bank_tilde *x, t_floatarg f){
    float g = (float)f;
    if (g < -1.f) g = -1.f;
    if (g >  1.f) g =  1.f;
    // set target; slewed value updated in perform
    x->fb_amt = g;
}

static void juicy_bank_tilde_fb_drive(t_juicy_bank_tilde *x, t_floatarg f){
    float d = (float)f;
    if (d < 0.f) d = 0.f;
    if (d > 1.f) d = 1.f;
    x->fb_drive = d;
}
static void juicy_bank_tilde_fb_timer(t_juicy_bank_tilde *x, t_floatarg f){
    float t = (float)f;
    if (t < 0.f) t = 0.f;
    if (t > 1.f) t = 1.f;
    x->fb_timer = t;
}
static void juicy_bank_tilde_fb_fmax(t_juicy_bank_tilde *x, t_floatarg f){
    float v = (float)f;
    if (v < 0.f) v = 0.f;
    if (v > 1.f) v = 1.f;
    x->fb_fmax = v;
}
// ---------- defaults helper ----------
static void jb_apply_default_saw(t_juicy_bank_tilde *x){
    x->n_modes = JB_MAX_MODES;
    x->edit_idx = 0;
    for(int i=0;i<JB_MAX_MODES;i++){
        x->base[i].active = 1;
        x->base[i].base_ratio = (float)(i+1);
        x->base[i].base_decay_ms = 1000.f;   // 1 second
        // True saw-like harmonic amplitude: ~1/n
        x->base[i].base_gain = 1.0f / (float)(i+1);
        x->base[i].attack_ms = 0.f;
        x->base[i].curve_amt = 0.f;          // linear
        x->base[i].pan = 0.f;
        x->base[i].keytrack = 1;
        x->base[i].disp_signature = 0.f;
        x->base[i].micro_sig      = 0.f;
    }
    // sensible body defaults
    x->damping = 0.f; x->brightness = 0.5f; x->position = 0.f; x->damp_broad=0.f; x->damp_point=0.f;
    x->density_amt = 0.f; x->density_mode = DENSITY_PIVOT;
    x->dispersion = 0.f; x->dispersion_last = -1.f;
    x->aniso = 0.f; x->aniso_eps = 0.02f;
    x->release_amt = 1.f;
}

// ---------- new() ----------
static void *juicy_bank_tilde_new(void){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)pd_new(juicy_bank_tilde_class);
    x->sr = sys_getsr(); if(x->sr<=0) x->sr=48000;

    // --- Startup spec (64 modes, real saw amplitude 1/n) ---
    jb_apply_default_saw(x);

    // body defaults
    x->damping=0.f; x->brightness=0.5f; x->position=0.f;
    x->density_amt=0.f; x->density_mode=DENSITY_PIVOT;
    x->dispersion=0.f; x->dispersion_last=-1.f;
    x->offset_amt=0.f;
    x->aniso=0.f; x->aniso_eps=0.02f;

    // Stretch default
    x->stretch = 0.f;

        x->warp = 0.f;
// realism defaults
    x->phase_rand=1.f; x->phase_debug=0;
    x->bandwidth=0.1f; x->micro_detune=0.1f;
    x->sine_pitch=0.f; x->sine_depth=0.f; x->sine_phase=0.f;

    // LFO + ADSR defaults
    x->lfo_shape = 1.f;   // default: shape 1 (for currently selected LFO)
    x->lfo_rate  = 1.f;   // 1 Hz
    x->lfo_phase = 0.f;   // start at phase 0
    x->lfo_index = 1.f;   // LFO 1 selected by default
    x->adsr_ms   = 0.f;   // ADSR env 0..1 (legacy name)

    // initialise per-LFO parameter and runtime state
    for (int li = 0; li < JB_N_LFO; ++li){
        x->lfo_shape_v[li]      = 1.f;
        x->lfo_rate_v[li]       = 1.f;
        x->lfo_phase_v[li]      = 0.f;
        x->lfo_phase_state[li]  = 0.f;
        x->lfo_val[li]          = 0.f;
        x->lfo_snh[li]          = 0.f;
    }

    // clear modulation matrix
    for(int i=0;i<JB_N_MODSRC;i++)
        for(int j=0;j<JB_N_MODTGT;j++)
            x->mod_matrix[i][j] = 0.f;


    // FEEDBACK defaults
    x->fb_amt   = 0.f;
    x->fb_amt_z = 0.f;
    x->fb_drive = 1.f;
    x->fb_timer = 0.f;
    x->fb_fmax  = 0.5f;
    x->basef0_ref=261.626f; // C4
    x->stiffen_amt=x->shortscale_amt=x->linger_amt=x->tilt_amt=x->bite_amt=x->bloom_amt=x->crossring_amt=0.f;

    x->max_voices = JB_MAX_VOICES;
    for(int v=0; v<JB_MAX_VOICES; ++v){
        x->v[v].state=V_IDLE; x->v[v].f0=x->basef0_ref; x->v[v].vel=0.f; x->v[v].energy=0.f;
        for(int i=0;i<JB_MAX_MODES;i++){
            x->v[v].disp_offset[i]=x->v[v].disp_target[i]=0.f;
            x->v[v].cr_gain_mul[i]=x->v[v].cr_decay_mul[i]=1.f;
        }
    }

    jb_rng_seed(&x->rng, 0xC0FFEEu);
    x->hp_a=0.f; x->hpL_x1=x->hpL_y1=x->hpR_x1=x->hpR_y1=0.f;

    x->exciter_mode = 0; // default legacy

    // INLETS (Signal → Behavior → Body → Individual)
    // Signal:
    CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy); // leftmost signal is implicit
    x->inR = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal); // main inR
    // per-voice pairs
    for(int i=0;i<JB_MAX_VOICES;i++){
        x->in_vL[i] = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
        x->in_vR[i] = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    }

    // Behavior (reduced)
    x->in_crossring  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("crossring"));
    x->in_release    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("release")); // release 0..1

    // Body (order: damping, brightness, position, density, dispersion, offset, anisotropy)
    x->in_damping    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damping"));
    x->in_damp_broad = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damp_broad"));
    x->in_damp_point = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damp_point"));
    x->in_brightness = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("brightness"));
    x->in_position   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("position"));
    x->in_density    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("density"));
    x->in_stretch    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("stretch"));
    x->in_warp       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("warp"));
    x->in_dispersion = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("dispersion"));
    x->in_offset     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("offset"));
    x->in_aniso      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("anisotropy"));
// SINE controls
    x->in_sine_pitch = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("sine_pitch"));
    x->in_sine_depth = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("sine_depth"));
    x->in_sine_phase = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("sine_phase"));

    // FEEDBACK controls (placed after sine_phase, before partials)
    x->in_fb_drive = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("fb_drive"));
    x->in_fb_amt   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("fb_amt"));
    x->in_fb_timer = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("fb_timer"));
    x->in_fb_fmax  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("fb_fmax"));
// Individual

    x->in_partials   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("partials"));
    x->in_index      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("index"));
    x->in_ratio      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("ratio"));
    x->in_gain       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("gain"));
    x->in_attack     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("attack"));
    x->in_decay      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("decay"));   // alias of decay
    x->in_curve      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("curve"));
    x->in_pan        = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("pan"));
    x->in_keytrack   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("keytrack"));

    // LFO + ADSR inlets (for future modulation matrix)
    x->in_lfo_shape = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("lfo_shape"));
    x->in_lfo_rate  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("lfo_rate"));
    x->in_lfo_phase = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("lfo_phase"));
    x->in_lfo_index = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("lfo_index"));
    // inlet for modulation-matrix configuration messages
    x->in_matrix    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("matrix"), gensym("matrix"));
    x->in_adsr_ms   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("adsr_ms"));

    // Outs
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);
// snapshot undo init
x->_undo_valid = 0;
for (int i=0;i<JB_MAX_MODES;i++){ x->_undo_base_gain[i]=0.f; x->_undo_base_decay_ms[i]=0.f; }

    x->out_index   = outlet_new(&x->x_obj, &s_float); // 1-based index reporter
    return (void *)x;
}


// ---------- INIT (factory re-init) ----------
static void juicy_bank_tilde_INIT(t_juicy_bank_tilde *x){
    // Apply 64-mode saw defaults (1/n amplitude), then reset states
    jb_apply_default_saw(x);
    juicy_bank_tilde_restart(x);
    post("juicy_bank~: INIT complete (64 modes, saw-like gains 1/n, decay=1s).");
}
static void juicy_bank_tilde_init_alias(t_juicy_bank_tilde *x){ juicy_bank_tilde_INIT(x); }

// ---------- setup ----------
// === Partials / Index navigation helpers ===
static void juicy_bank_tilde_partials(t_juicy_bank_tilde *x, t_floatarg f){
    int K = (int)floorf(f + 0.5f);
    if (K < 0) K = 0;
    if (K > x->n_modes) K = x->n_modes;
    x->active_modes = K;
    // Clamp current edit index to active range (if any)
    if (x->active_modes == 0) {
        x->edit_idx = 0;
    } else if (x->edit_idx >= x->active_modes) {
        x->edit_idx = x->active_modes - 1;
    }
}

static void juicy_bank_tilde_index_forward(t_juicy_bank_tilde *x){
    int K = (x->active_modes > 0) ? x->active_modes : 1;
    x->edit_idx = (x->edit_idx + 1) % K;
    if (x->out_index) outlet_float(x->out_index, (t_float)(x->edit_idx + 1));
}

static void juicy_bank_tilde_index_backward(t_juicy_bank_tilde *x){
    int K = (x->active_modes > 0) ? x->active_modes : 1;
    x->edit_idx = (x->edit_idx - 1 + K) % K;
    if (x->out_index) outlet_float(x->out_index, (t_float)(x->edit_idx + 1));
}


// ---------- SNAPSHOT: bake current SINE mask into base gains and DAMPER into base decays ----------
static void juicy_bank_tilde_snapshot(t_juicy_bank_tilde *x){
    // Save undo of base fields
    for (int i=0;i<JB_MAX_MODES;i++){
        x->_undo_base_gain[i] = x->base[i].base_gain;
        x->_undo_base_decay_ms[i] = x->base[i].base_decay_ms;
    }
    x->_undo_valid = 1;

    // Precompute SINE AM mask per index (same math as in jb_update_voice_gains)
    int N = x->n_modes;
    float pitch = jb_clamp(x->sine_pitch, 0.f, 1.f);
    float depth = jb_clamp(x->sine_depth, -1.f, 1.f);
    float phase = x->sine_phase;
    float cycles_min = 0.25f;
    float cycles_max = floorf((float)((N>0)?N:1) * 0.5f);
    if (cycles_max < cycles_min) cycles_max = cycles_min;
    float cycles = cycles_min + pitch * (cycles_max - cycles_min);

    // Damper weights per mode (same formula as in jb_update_voice_coeffs)
    float b = jb_clamp(x->damp_broad, 0.f, 1.f);
    float p = x->damp_point; if (p < 0.f) p = 0.f;
            if (p > 1.f) p = 1.f;
    float n = (float)((x->n_modes>0)?x->n_modes:1);
    float sigma_min = 0.5f / n;      // ~single-mode width
    float sigma_max = 0.5f;          // whole bank
    float sigma = (1.f - b)*sigma_max + b*sigma_min;
    float d_amt_global = jb_clamp(x->damping, -1.f, 1.f);

    for(int i=0;i<x->n_modes;i++){
        if (!x->base[i].active) continue;
        // --- SINE mask (bipolar, same semantics as jb_update_voice_gains) ---
        float k_norm = (N>1) ? ((float)i / (float)(N-1)) : 0.f;
        float theta = 2.f * (float)M_PI * (cycles * k_norm + phase);
        float w01 = 0.5f * (1.f + cosf(theta)); // 1 at pattern center, 0 at pattern nulls
        float sharp = 1.0f + 8.0f * fabsf(depth);   // use |depth| for window sharpness
        float pattern = powf(w01, sharp);          // pattern membership mask in [0,1]

        // bipolar amount:
        //   depth > 0: attenuate pattern partials
        //   depth < 0: attenuate non-pattern partials
        float a_pos = (depth > 0.f) ? depth : 0.f;
        float a_neg = (depth < 0.f) ? -depth : 0.f;

        float weight = 1.f
                       - a_pos * pattern          // turn down pattern when depth > 0
                       - a_neg * (1.f - pattern); // turn down complement when depth < 0;

        if (weight < 0.f) weight = 0.f;
        x->base[i].base_gain *= weight;


// --- DAMPER bake into base_decay_ms ---
// mirror jb_update_voice_coeffs damping logic onto the stored T60
        float k_mode = (x->n_modes>1)? ((float)i/(float)(x->n_modes-1)) : 0.f;
        float dx = fabsf(k_mode - p); if (dx > 0.5f) dx = 1.f - dx; // circular distance
        float wloc = expf(-0.5f * (dx*dx) / (sigma*sigma)); // 0..1
        float d_amt = d_amt_global * wloc;                  // local -1..1
        // convert current base decay to seconds and apply same mapping as runtime
        float T60 = jb_clamp(x->base[i].base_decay_ms, 0.f, 1e7f) * 0.001f;
        if (d_amt >= 0.f){
            T60 *= (1.f - d_amt);
        } else {
            float Dneg = -d_amt;
            float ceiling = JB_MAX_DECAY_S;
            T60 = T60 + Dneg * (ceiling - T60);
            if (T60 > ceiling) T60 = ceiling;
        }
        x->base[i].base_decay_ms = T60 * 1000.f;
        if (x->base[i].base_decay_ms < 0.f) x->base[i].base_decay_ms = 0.f;
    }

    // Neutralize the two controllers so you can re-apply on top of the baked shape
    x->sine_depth = 0.f;
    x->damping = 0.f;
}

static void juicy_bank_tilde_snapshot_undo(t_juicy_bank_tilde *x){
    if (!x->_undo_valid) return;
    for (int i=0;i<JB_MAX_MODES;i++){
        x->base[i].base_gain = x->_undo_base_gain[i];
        x->base[i].base_decay_ms = x->_undo_base_decay_ms[i];
    }
    x->_undo_valid = 0;
}

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
    else if (!strcmp(src, "adsr"))     src_idx = 2;
    else if (!strcmp(src, "lfo1"))     src_idx = 3;
    else if (!strcmp(src, "lfo2"))     src_idx = 4;
    else return 0;

    // --- targets ---
    if (!strcmp(tgt, "damping") || !strcmp(tgt, "damper"))              tgt_idx = 0;
    else if (!strcmp(tgt, "broadness"))                                 tgt_idx = 1;
    else if (!strcmp(tgt, "location"))                                  tgt_idx = 2;
    else if (!strcmp(tgt, "brightness"))                                tgt_idx = 3;
    else if (!strcmp(tgt, "position"))                                  tgt_idx = 4;
    else if (!strcmp(tgt, "density"))                                   tgt_idx = 5;
    else if (!strcmp(tgt, "stretch"))                                   tgt_idx = 6;
    else if (!strcmp(tgt, "warp"))                                      tgt_idx = 7;
    else if (!strcmp(tgt, "offset"))                                    tgt_idx = 8;
    else if (!strcmp(tgt, "sinepattern") || !strcmp(tgt, "sine_pattern")) tgt_idx = 9;
    else if (!strcmp(tgt, "sinephase")   || !strcmp(tgt, "sine_phase"))   tgt_idx = 10;
    else if (!strcmp(tgt, "master"))                                    tgt_idx = 11;
    else if (!strcmp(tgt, "pitch"))                                     tgt_idx = 12;
    else if (!strcmp(tgt, "pan"))                                       tgt_idx = 13;
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
static void juicy_bank_tilde_matrix(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    if (argc < 2) return;
    if (argv[0].a_type != A_SYMBOL) return;

    t_symbol *route = atom_getsymbol(argv);
    if (!route) return;

    int src_idx = -1, tgt_idx = -1;
    if (!jb_modmatrix_parse_selector(route->s_name, &src_idx, &tgt_idx))
        return;

    t_float amt = atom_getfloat(argv + 1);
    if (amt < -1.f) amt = -1.f;
    else if (amt > 1.f) amt = 1.f;

    if (src_idx >= 0 && src_idx < JB_N_MODSRC &&
        tgt_idx >= 0 && tgt_idx < JB_N_MODTGT){
        x->mod_matrix[src_idx][tgt_idx] = amt;
    }
}

void juicy_bank_tilde_setup(void){
    juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
                           (t_newmethod)juicy_bank_tilde_new,
                           (t_method)juicy_bank_tilde_free,
                           sizeof(t_juicy_bank_tilde), CLASS_DEFAULT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_snapshot, gensym("snapshot"), 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_snapshot_undo, gensym("snapshot_undo"), 0);
    CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy);

    // accept modulation-matrix configuration messages in two formats:
    // 1) Direct: "lfo1_to_pitch 0.5" (left inlet, via 'anything')
    // 2) Tagged: "matrix lfo1_to_pitch 0.5" (matrix inlet, via 'matrix' method)
    class_addanything(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anything);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_matrix, gensym("matrix"), A_GIMME, 0);

    // BEHAVIOR
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_stiffen, gensym("stiffen"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_shortscale, gensym("shortscale"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_shortscale, gensym("shortscle"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_linger, gensym("linger"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_tilt, gensym("tilt"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bite, gensym("bite"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bloom, gensym("bloom"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_crossring, gensym("crossring"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_release, gensym("release"), A_DEFFLOAT, 0);

    // BODY
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damping, gensym("damping"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damp_broad, gensym("damp_broad"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damp_point, gensym("damp_point"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position, gensym("position"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density, gensym("density"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_offset, gensym("offset"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_aniso_eps, gensym("aniso_epsilon"), A_DEFFLOAT, 0);

    
    
    
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_stretch, gensym("stretch"), A_FLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_warp, gensym("warp"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_release, gensym("release"), A_DEFFLOAT, 0);
// SINE methods
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_sine_pitch, gensym("sine_pitch"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_sine_depth, gensym("sine_depth"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_sine_phase, gensym("sine_phase"), A_DEFFLOAT, 0);

    // LFO + ADSR methods
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_shape, gensym("lfo_shape"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_rate,  gensym("lfo_rate"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_phase, gensym("lfo_phase"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_lfo_index, gensym("lfo_index"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_adsr_ms,   gensym("adsr_ms"),   A_DEFFLOAT, 0);
// INDIVIDUAL (per-mode)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index, gensym("index"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_ratio_i, gensym("ratio"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_gain_i, gensym("gain"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_attack_i, gensym("attack"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay_i, gensym("decay"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay_i, gensym("decya"), A_DEFFLOAT, 0); // alias
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_curve_i, gensym("curve"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pan_i, gensym("pan"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_keytrack_i, gensym("keytrack"), A_DEFFLOAT, 0);

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
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_random, gensym("phase_random"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_debug, gensym("phase_debug"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bandwidth, gensym("bandwidth"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_micro_detune, gensym("micro_detune"), A_DEFFLOAT, 0);

    // notes/poly (non-voice-specific)
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note, gensym("note"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_midi, gensym("note_midi"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_off, gensym("off"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_voices, gensym("voices"), A_DEFFLOAT, 0);

    // voice-addressed (for [poly])
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_poly, gensym("note_poly"), A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_note_poly_midi, gensym("note_poly_midi"), A_DEFFLOAT, A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_off_poly, gensym("off_poly"), A_DEFFLOAT, 0);

    // base & reset
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_basef0, gensym("basef0"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_base_alias, gensym("base"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_exciter_mode, gensym("exciter_mode"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_restart, gensym("restart"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_INIT, gensym("INIT"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_init_alias, gensym("init"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_partials, gensym("partials"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index_forward, gensym("forward"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_index_backward, gensym("backward"), 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_stretch, gensym("stretch"), A_FLOAT, 0);

        class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_fb_amt,   gensym("fb_amt"),   A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_fb_drive,   gensym("fb_drive"),   A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_fb_timer,   gensym("fb_timer"),   A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_fb_fmax,    gensym("fb_fmax"),    A_DEFFLOAT, 0);
}
