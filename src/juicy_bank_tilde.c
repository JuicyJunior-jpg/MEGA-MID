
// juicy_bank_tilde.c
// "juicy_bank~" — Modal resonator bank with stereo outs and full inlet control.
// Build (Linux):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//      -I"/usr/include/pd" -shared -fPIC -Wl,-export-dynamic -lm \
//      -o juicy_bank~.pd_linux juicy_bank_tilde.c
// Build (macOS universal):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//      -I"/Applications/Pd-0.56-1.app/Contents/Resources/src" \
//      -arch arm64 -arch x86_64 -mmacosx-version-min=10.13 \
//      -bundle -undefined dynamic_lookup \
//      -o juicy_bank~.pd_darwin juicy_bank_tilde.c
//
// © 2025 Juicy x ChatGPT

#include "m_pd.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define JB_MAX_MODES 20

// ---------- helpers ----------
static inline float jb_clamp(float x, float lo, float hi){ return x < lo ? lo : (x > hi ? hi : x); }
static inline float jb_mmix(float a, float b, float t){ return a + (b - a) * t; }

// RNG (xorshift32)
typedef struct { uint32_t s; } jb_rng_t;
static inline void jb_rng_seed(jb_rng_t* r, uint32_t s){ if(s==0) s=1u; r->s = s; }
static inline uint32_t jb_rng_u32(jb_rng_t* r){ uint32_t x=r->s; x ^= x<<13; x ^= x>>17; x ^= x<<5; return r->s = x; }
static inline float jb_rng_bi(jb_rng_t* r){ return ((jb_rng_u32(r) / 4294967296.0f) * 2.f) - 1.f; } // [-1,1)

typedef enum {
    DENSITY_PIVOT = 0,
    DENSITY_INDIVIDUAL = 1
} density_mode_e;

// ---------- per-mode ----------
typedef struct {
    int   active;
    float base_ratio;     // user "ratio" OR absolute Hz if you feed freq list
    float base_decay_ms;  // ms
    float base_gain;      // 0..1
    float curve_amt;      // -1..1 (stored, hook for shaping later)
    float attack_ms;      // input drive attack smoothing
    float pan;            // -1..1

    // derived per-block
    float ratio_now;      // after density
    float decay_ms_now;   // after damping
    float gain_now;       // after brightness/pos/aniso

    // reson state
    float a1, a2;
    float y1, y2;
    float drive;          // smoothed drive input

    // contact env follower
    float env;

    // dispersion
    float disp_signature; // stable [-1,1]
} jb_mode_t;

// ---------- object ----------
typedef struct _juicy_bank_tilde {
    t_object  x_obj;
    t_float   f_dummy;      // CLASS_MAINSIGNALIN

    // outlets
    t_outlet *outL, *outR;

    // extra signal inlet (right audio)
    t_inlet *in_audioR_sig;

    // float inlets (left→right after signals)
    t_inlet *in_damp, *in_bright, *in_pos, *in_disp, *in_density, *in_aniso, *in_contact;
    t_inlet *in_index, *in_ratio, *in_gain, *in_attack, *in_decay, *in_curve, *in_pan, *in_keytrack;

    // per-index float inlet storage (NaN = "unset")
    float f_ratio_in, f_gain_in, f_attack_in, f_decay_in, f_curve_in, f_pan_in;

    // params
    float sr;
    int   n_modes;
    jb_mode_t m[JB_MAX_MODES];
    int fundamental_idx;

    // globals
    float damping;       // 0..1
    float brightness;    // 0..1 (0 mutes highs, 0.5 neutral, 1 boosts highs)
    float position;      // 0..1 (0 bypass)
    float dispersion;    // 0..1
    float density;       // -1..1
    density_mode_e density_mode;
    float anisotropy;    // -1..1
    float contact_amt;   // 0..1
    float contact_sym;   // -1..1
    float keytrack_amount; // 0..1 amount
    int   keytrack_on;   // 0/1
    float basef0;        // Hz used when keytrack_on

    // UI index
    t_float sel_index_f; // 1-based
    int     sel_index;   // cached 0-based

    // tuneables
    float brightness_alpha;  // slope for tilt
    float env_follower_tau;  // contact env time
    float contact_threshold; // small threshold
    float epsilon_near_int;  // aniso integer closeness

    // dispersion smoothing
    jb_rng_t rng;
    uint32_t seed_value;
    float disp_offset[JB_MAX_MODES];
    float disp_target[JB_MAX_MODES];
    float dispersion_slew; // 0..1 per-block

} t_juicy_bank_tilde;

static t_class *juicy_bank_class;

// ---------- utils ----------
static int jb_is_near_integer(float x, float eps){
    float r = roundf(x);
    return (fabsf(x - r) <= eps);
}

static int jb_find_fundamental(t_juicy_bank_tilde* x){
    int best = -1; float bestd = 1e9f;
    for (int i=0;i<x->n_modes;i++){
        if (!x->m[i].active) continue;
        float d = fabsf(x->m[i].base_ratio - 1.f);
        if (d < bestd){ bestd = d; best = i; }
    }
    return best;
}

static void jb_apply_density(t_juicy_bank_tilde* x){
    float s = 1.f + 0.5f * x->density; // -1..1 => 0.5..1.5
    if (x->density_mode == DENSITY_PIVOT){
        for (int i=0;i<x->n_modes;i++){
            jb_mode_t* md = &x->m[i];
            if (!md->active){ md->ratio_now = md->base_ratio; continue; }
            if (i == x->fundamental_idx){ md->ratio_now = 1.f; continue; }
            md->ratio_now = 1.f + (md->base_ratio - 1.f) * s;
        }
    } else {
        // individual chain spacing based on previous ACTIVE mode, preserving order
        int idxs[JB_MAX_MODES]; int count=0;
        for (int i=0;i<x->n_modes;i++) if (x->m[i].active) idxs[count++] = i;
        for (int a=1;a<count;a++){
            int j=a, id=idxs[a];
            while(j>0 && x->m[idxs[j-1]].base_ratio > x->m[id].base_ratio){ idxs[j]=idxs[j-1]; j--; }
            idxs[j]=id;
        }
        int first = -1;
        for (int k=0;k<count;k++){
            int i = idxs[k];
            jb_mode_t* md = &x->m[i];
            if (first < 0){
                md->ratio_now = md->base_ratio;
                first = i;
            } else {
                int prev_i = idxs[k-1];
                jb_mode_t* pm = &x->m[prev_i];
                float d = (md->base_ratio - x->m[prev_i].base_ratio) * s;
                md->ratio_now = pm->ratio_now + d;
            }
        }
        for (int i=0;i<x->n_modes;i++) if (!x->m[i].active) x->m[i].ratio_now = x->m[i].base_ratio;
    }
}

static void jb_reroll_dispersion_targets(t_juicy_bank_tilde* x){
    for (int i=0;i<x->n_modes;i++){
        jb_mode_t* md = &x->m[i];
        if (!md->active || i == x->fundamental_idx){ x->disp_target[i] = 0.f; continue; }
        x->disp_target[i] = md->disp_signature * x->dispersion; // ± up to 1.0
    }
}

static void jb_slew_dispersion(t_juicy_bank_tilde* x){
    float a = x->dispersion_slew;
    for (int i=0;i<x->n_modes;i++){
        x->disp_offset[i] = jb_mmix(x->disp_offset[i], x->disp_target[i], a);
    }
}

static void jb_update_per_mode_gains(t_juicy_bank_tilde* x){
    float b = x->brightness;            // 0..1
    float t = 2.f*(b - 0.5f);           // -1..+1
    float alpha = x->brightness_alpha;  // slope
    float eps = x->epsilon_near_int;
    float pos = x->position;
    int use_pos = (pos > 0.f);

    for (int i=0;i<x->n_modes;i++){
        jb_mode_t* md = &x->m[i];
        if (!md->active){ md->gain_now = 0.f; continue; }

        float g = md->base_gain;

        float r = md->ratio_now + x->disp_offset[i];
        if (r < 0.0001f) r = 0.0001f;
        float bright_fac = powf(r, alpha * t);
        if (i == x->fundamental_idx) bright_fac = 1.f;
        g *= bright_fac;

        int is_even = jb_is_near_integer(r, eps);
        float a = x->anisotropy; // -1..1
        float w = 1.f;
        if (i != x->fundamental_idx){
            if (a > 0.f){ // favor even -> mute "odd"
                if (!is_even){ float k = jb_clamp(a,0,1); w *= (1.f - k); }
            } else if (a < 0.f){ // favor odd -> mute "even"
                if (is_even){ float k = jb_clamp(-a,0,1); w *= (1.f - k); }
            }
        }
        g *= jb_clamp(w, 0.f, 1.f);

        if (use_pos){
            int k = (int)floorf(r + 0.5f);
            if (k < 1) k = 1;
            float pw = fabsf(sinf(M_PI * (float)k * pos)); // true nodes
            g *= pw;
        }

        md->gain_now = g;
    }
}

static void jb_update_coeffs(t_juicy_bank_tilde* x){
    float sr = x->sr;
    for (int i=0;i<x->n_modes;i++){
        jb_mode_t* md = &x->m[i];
        if (!md->active){ md->a1 = md->a2 = md->y1 = md->y2 = 0.f; continue; }
        float ratio = md->ratio_now + x->disp_offset[i];
        if (ratio < 0.0001f) ratio = 0.0001f;

        // keytrack blend: 0 => absolute Hz (ratio interpreted as Hz), 1 => basef0 * ratio
        float f_abs = ratio;
        float f_trk = (x->basef0 > 0.f) ? (x->basef0 * ratio) : ratio;
        float kamt = (x->keytrack_on ? jb_clamp(x->keytrack_amount, 0.f, 1.f) : 0.f);
        float f = f_abs * (1.f - kamt) + f_trk * kamt;

        float w = 2.f * M_PI * (f / sr);
        if (w > M_PI) w = M_PI;

        float T60 = jb_clamp(md->decay_ms_now, 0.f, 1e6f) * 0.001f;
        float r = (T60 <= 0.f) ? 0.f : powf(10.f, -3.f / (T60 * sr));

        float c = cosf(w);
        md->a1 = 2.f * r * c;
        md->a2 = -r * r;
    }
}

static inline float jb_contact_shape(float y, float amt, float sym){
    if (amt <= 0.f) return y;
    float sign_bias = sym * 0.5f;
    float z = y + sign_bias * y * y;
    float drive = 1.f + 10.f * amt;
    float shaped = tanhf(drive * z);
    float makeup = (drive > 0.f) ? (1.f / drive) : 1.f;
    return shaped * makeup;
}

static inline void jb_sanitize_globals(t_juicy_bank_tilde* x){
    x->damping     = jb_clamp(x->damping, 0.f, 1.f);
    x->brightness  = jb_clamp(x->brightness, 0.f, 1.f);
    x->position    = jb_clamp(x->position, 0.f, 1.f);
    x->dispersion  = jb_clamp(x->dispersion, 0.f, 1.f);
    x->density     = jb_clamp(x->density, -1.f, 1.f);
    x->anisotropy  = jb_clamp(x->anisotropy, -1.f, 1.f);
    x->contact_amt = jb_clamp(x->contact_amt, 0.f, 1.f);
    x->contact_sym = jb_clamp(x->contact_sym, -1.f, 1.f);
    x->keytrack_amount = jb_clamp(x->keytrack_amount, 0.f, 1.f);
}

// ---------- perform ----------
static t_int *juicy_bank_tilde_perform(t_int *w){
    t_juicy_bank_tilde *x = (t_juicy_bank_tilde *)(w[1]);
    t_sample *inL = (t_sample *)(w[2]);
    t_sample *inR = (t_sample *)(w[3]);
    t_sample *outL = (t_sample *)(w[4]);
    t_sample *outR = (t_sample *)(w[5]);
    int n = (int)(w[6]);

    // clamp globals
    jb_sanitize_globals(x);

    // apply per-index writes from float inlets (NaN = no change)
    x->sel_index = ((int)x->sel_index_f) - 1;
    if (x->sel_index >= 0 && x->sel_index < x->n_modes){
        jb_mode_t* md = &x->m[x->sel_index];
        if (md->active){
            if (!isnan(x->f_ratio_in)){ md->base_ratio    = (x->f_ratio_in >= 0.f ? x->f_ratio_in : md->base_ratio); x->f_ratio_in  = NAN; }
            if (!isnan(x->f_gain_in)) { md->base_gain     = jb_clamp(x->f_gain_in, 0.f, 1.f);                       x->f_gain_in   = NAN; }
            if (!isnan(x->f_attack_in)){md->attack_ms     = jb_clamp(x->f_attack_in, 0.f, 5000.f);                  x->f_attack_in = NAN; }
            if (!isnan(x->f_decay_in)){ md->base_decay_ms = (x->f_decay_in >= 0.f ? x->f_decay_in : md->base_decay_ms); x->f_decay_in  = NAN; }
            if (!isnan(x->f_curve_in)){ md->curve_amt     = jb_clamp(x->f_curve_in, -1.f, 1.f);                     x->f_curve_in  = NAN; }
            if (!isnan(x->f_pan_in))  { md->pan           = jb_clamp(x->f_pan_in, -1.f, 1.f);                       x->f_pan_in    = NAN; }
        }
    }

    // derived mappings
    x->fundamental_idx = jb_find_fundamental(x);
    jb_apply_density(x);

    // update dispersion targets & slew toward them
    jb_reroll_dispersion_targets(x);
    jb_slew_dispersion(x);

    // apply damping to get per-mode decay_now
    for (int mi=0; mi<x->n_modes; ++mi){
        jb_mode_t* md = &x->m[mi];
        md->decay_ms_now = md->base_decay_ms * (1.f - x->damping);
    }

    // update gains & coeffs
    jb_update_per_mode_gains(x);
    jb_update_coeffs(x);

    float env_tau = x->env_follower_tau;
    float th  = x->contact_threshold;
    float camt = x->contact_amt;
    float csym = x->contact_sym;

    for (int i=0;i<n;i++){
        float exc = inL[i] + inR[i]; // sum excitation

        float sumL = 0.f, sumR = 0.f;
        for (int m=0;m<x->n_modes;m++){
            jb_mode_t* md = &x->m[m];
            if (!md->active || md->gain_now <= 0.f) continue;

            // attack smoothing on drive
            float att_samps = jb_clamp(md->attack_ms, 0.f, 5000.f) * 0.001f * x->sr;
            float att_a = (att_samps > 0.f) ? (1.f - expf(-1.f / (att_samps))) : 1.f;
            float drive_target = md->gain_now * exc;
            md->drive += att_a * (drive_target - md->drive);

            // reson
            float y = (md->a1 * md->y1 + md->a2 * md->y2) + md->drive;
            md->y2 = md->y1;
            md->y1 = y;

            // contact
            float abs_y = fabsf(y);
            md->env = md->env + env_tau * (abs_y - md->env);
            if (camt > 0.f && md->env > th){
                y = jb_contact_shape(y, camt, csym);
            }

            // equal-power pan
            float pan = jb_clamp(md->pan, -1.f, 1.f);
            float theta = (pan + 1.f) * (M_PI * 0.25f); // -1..1 -> 0..pi/2
            float gl = cosf(theta);
            float gr = sinf(theta);

            sumL += y * gl;
            sumR += y * gr;
        }

        outL[i] = sumL;
        outR[i] = sumR;
    }

    return (w + 7);
}

// ---------- dsp glue ----------
static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;
    dsp_add(juicy_bank_tilde_perform, 6,
            x, sp[0]->s_vec, // inL
            sp[1]->s_vec,    // inR
            sp[2]->s_vec,    // outL
            sp[3]->s_vec,    // outR
            sp[0]->s_n);
}

// ---------- setters / messages ----------
static void juicy_bank_tilde_modes(t_juicy_bank_tilde *x, t_floatarg f){
    int nm = (int)f; if (nm < 1) nm = 1; if (nm > JB_MAX_MODES) nm = JB_MAX_MODES;
    x->n_modes = nm;
}
static void juicy_bank_tilde_active(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg onf){
    int idx = (int)idxf - 1; if (idx < 0 || idx >= x->n_modes) return;
    x->m[idx].active = (onf != 0.f);
    // assign a signature deterministically from seed+index
    jb_rng_t tmp = x->rng; for (int k=0;k<=idx;k++) jb_rng_u32(&tmp);
    x->m[idx].disp_signature = jb_rng_bi(&tmp);
    x->fundamental_idx = jb_find_fundamental(x);
}
static void juicy_bank_tilde_ratio_msg(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg ratio){
    int idx = (int)idxf - 1; if (idx < 0 || idx >= x->n_modes) return;
    if (ratio < 0.f) ratio = 0.f;
    x->m[idx].base_ratio = ratio;
    x->m[idx].ratio_now  = ratio;
    x->fundamental_idx = jb_find_fundamental(x);
}
static void juicy_bank_tilde_decay_msg(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg ms){
    int idx = (int)idxf - 1; if (idx < 0 || idx >= x->n_modes) return;
    if (ms < 0.f) ms = 0.f;
    x->m[idx].base_decay_ms = ms;
}
static void juicy_bank_tilde_gain_msg(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg g){
    int idx = (int)idxf - 1; if (idx < 0 || idx >= x->n_modes) return;
    x->m[idx].base_gain = jb_clamp(g, 0.f, 1.f);
}

// batch lists
static void juicy_bank_tilde_freqlist(t_juicy_bank_tilde *x, t_symbol* s, int argc, t_atom* argv){
    int n = (argc < x->n_modes)? argc : x->n_modes;
    for (int i=0;i<n;i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i);
            if (v < 0.f) v = 0.f;
            x->m[i].base_ratio = v; // as absolute Hz if you want
            x->m[i].ratio_now  = v;
        }
    }
    x->fundamental_idx = jb_find_fundamental(x);
}
static void juicy_bank_tilde_decaylist(t_juicy_bank_tilde *x, t_symbol* s, int argc, t_atom* argv){
    int n = (argc < x->n_modes)? argc : x->n_modes;
    for (int i=0;i<n;i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i); if (v < 0.f) v = 0.f;
            x->m[i].base_decay_ms = v;
        }
    }
}
static void juicy_bank_tilde_amplist(t_juicy_bank_tilde *x, t_symbol* s, int argc, t_atom* argv){
    int n = (argc < x->n_modes)? argc : x->n_modes;
    for (int i=0;i<n;i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv+i);
            x->m[i].base_gain = jb_clamp(v, 0.f, 1.f);
        }
    }
}

// global messages
static void juicy_bank_tilde_damping(t_juicy_bank_tilde *x, t_floatarg f){ x->damping = jb_clamp(f,0,1); }
static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness = jb_clamp(f,0,1); }
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){ x->position = jb_clamp(f,0,1); }
static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){ x->dispersion = jb_clamp(f,0,1); jb_reroll_dispersion_targets(x); }
static void juicy_bank_tilde_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->anisotropy = jb_clamp(f,-1,1); }
static void juicy_bank_tilde_contact(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_amt = jb_clamp(f,0,1); }
static void juicy_bank_tilde_contact_sym(t_juicy_bank_tilde *x, t_floatarg f){
    if (f < -1.f) f = -1.f;
    if (f >  1.f) f =  1.f;
    x->contact_sym = f;
}
static void juicy_bank_tilde_density_msg(t_juicy_bank_tilde *x, t_floatarg f){ x->density = jb_clamp(f,-1,1); jb_apply_density(x); jb_reroll_dispersion_targets(x); }
static void juicy_bank_tilde_density_pivot(t_juicy_bank_tilde *x){ x->density_mode = DENSITY_PIVOT; jb_apply_density(x); jb_reroll_dispersion_targets(x); }
static void juicy_bank_tilde_density_individual(t_juicy_bank_tilde *x){ x->density_mode = DENSITY_INDIVIDUAL; jb_apply_density(x); jb_reroll_dispersion_targets(x); }

static void juicy_bank_tilde_seed(t_juicy_bank_tilde *x, t_floatarg f){
    uint32_t s = (uint32_t)(f < 0 ? -f : f); if (s==0) s=1u;
    x->seed_value = s; jb_rng_seed(&x->rng, s);
    for (int i=0;i<x->n_modes;i++){
        jb_rng_t tmp = x->rng; for (int k=0;k<=i;k++) jb_rng_u32(&tmp);
        x->m[i].disp_signature = jb_rng_bi(&tmp);
    }
    jb_reroll_dispersion_targets(x);
}
static void juicy_bank_tilde_dispersion_reroll(t_juicy_bank_tilde *x){
    x->seed_value = x->seed_value * 1664525u + 1013904223u;
    jb_rng_seed(&x->rng, x->seed_value);
    for (int i=0;i<x->n_modes;i++){
        jb_rng_t tmp = x->rng; for (int k=0;k<=i;k++) jb_rng_u32(&tmp);
        x->m[i].disp_signature = jb_rng_bi(&tmp);
    }
    jb_reroll_dispersion_targets(x);
}
static void juicy_bank_tilde_aniso_epsilon(t_juicy_bank_tilde *x, t_floatarg f){
    float e = f; if (e < 0.f) e = 0.f; if (e > 0.25f) e = 0.25f; x->epsilon_near_int = e;
}

static void juicy_bank_tilde_keytrack_on(t_juicy_bank_tilde *x, t_floatarg f){ x->keytrack_on = (f != 0.f); }
static void juicy_bank_tilde_basef0(t_juicy_bank_tilde *x, t_floatarg f){ x->basef0 = (f > 0.f ? f : 0.f); }

// ---------- new/free ----------
static void *juicy_bank_tilde_new(t_symbol *s, int argc, t_atom *argv){
    t_juicy_bank_tilde *x = (t_juicy_bank_tilde *)pd_new(juicy_bank_class);

    // outlets
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);
    // extra signal inlet (right)
    x->in_audioR_sig = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);

    // float inlets (damping..keytrack)
    x->in_damp     = floatinlet_new(&x->x_obj, &x->damping);
    x->in_bright   = floatinlet_new(&x->x_obj, &x->brightness);
    x->in_pos      = floatinlet_new(&x->x_obj, &x->position);
    x->in_disp     = floatinlet_new(&x->x_obj, &x->dispersion);
    x->in_density  = floatinlet_new(&x->x_obj, &x->density);
    x->in_aniso    = floatinlet_new(&x->x_obj, &x->anisotropy);
    x->in_contact  = floatinlet_new(&x->x_obj, &x->contact_amt);
    x->in_index    = floatinlet_new(&x->x_obj, &x->sel_index_f);
    x->in_ratio    = floatinlet_new(&x->x_obj, &x->f_ratio_in);
    x->in_gain     = floatinlet_new(&x->x_obj, &x->f_gain_in);
    x->in_attack   = floatinlet_new(&x->x_obj, &x->f_attack_in);
    x->in_decay    = floatinlet_new(&x->x_obj, &x->f_decay_in);
    x->in_curve    = floatinlet_new(&x->x_obj, &x->f_curve_in);
    x->in_pan      = floatinlet_new(&x->x_obj, &x->f_pan_in);
    x->in_keytrack = floatinlet_new(&x->x_obj, &x->keytrack_amount);

    // defaults
    x->sr = sys_getsr(); if (x->sr <= 0) x->sr = 48000.f;
    x->n_modes = JB_MAX_MODES;
    for (int i=0;i<JB_MAX_MODES;i++){
        jb_mode_t* md = &x->m[i];
        md->active = (i < 8);
        md->base_ratio = (float)(i+1);
        md->base_decay_ms = 500.f;
        md->base_gain = 0.2f;
        md->curve_amt = 0.f;
        md->attack_ms = 0.f;
        md->pan = 0.f;
        md->ratio_now = md->base_ratio;
        md->decay_ms_now = md->base_decay_ms;
        md->gain_now = md->base_gain;
        md->a1 = md->a2 = md->y1 = md->y2 = md->drive = 0.f;
        md->env = 0.f;
        md->disp_signature = 0.f;
    }
    x->fundamental_idx = jb_find_fundamental(x);

    x->damping = 0.f;
    x->brightness = 0.5f;
    x->position = 0.f;
    x->dispersion = 0.f;
    x->density = 0.f;
    x->density_mode = DENSITY_PIVOT;
    x->anisotropy = 0.f;
    x->contact_amt = 0.f;
    x->contact_sym = 0.f;
    x->keytrack_amount = 0.f; x->keytrack_on = 0; x->basef0 = 0.f;

    x->sel_index_f = 1.f; x->sel_index = 0;

    x->brightness_alpha = 0.75f;
    x->env_follower_tau = 0.01f;
    x->contact_threshold = 0.02f;
    x->epsilon_near_int = 0.02f;

    jb_rng_seed(&x->rng, 1234567u);
    x->seed_value = 1234567u;
    for (int i=0;i<JB_MAX_MODES;i++){
        jb_rng_t tmp = x->rng; for (int k=0;k<=i;k++) jb_rng_u32(&tmp);
        x->m[i].disp_signature = jb_rng_bi(&tmp);
        x->disp_offset[i] = 0.f;
        x->disp_target[i] = 0.f;
    }
    x->dispersion_slew = 0.05f;

    // per-index inlet latches start as NaN (so 0 is a valid set value)
    x->f_ratio_in = x->f_gain_in = x->f_attack_in = x->f_decay_in = x->f_curve_in = x->f_pan_in = NAN;

    return (void *)x;
}

static void juicy_bank_tilde_free(t_juicy_bank_tilde *x){
    if (x->in_audioR_sig) inlet_free(x->in_audioR_sig);
    if (x->in_damp) inlet_free(x->in_damp);
    if (x->in_bright) inlet_free(x->in_bright);
    if (x->in_pos) inlet_free(x->in_pos);
    if (x->in_disp) inlet_free(x->in_disp);
    if (x->in_density) inlet_free(x->in_density);
    if (x->in_aniso) inlet_free(x->in_aniso);
    if (x->in_contact) inlet_free(x->in_contact);
    if (x->in_index) inlet_free(x->in_index);
    if (x->in_ratio) inlet_free(x->in_ratio);
    if (x->in_gain) inlet_free(x->in_gain);
    if (x->in_attack) inlet_free(x->in_attack);
    if (x->in_decay) inlet_free(x->in_decay);
    if (x->in_curve) inlet_free(x->in_curve);
    if (x->in_pan) inlet_free(x->in_pan);
    if (x->in_keytrack) inlet_free(x->in_keytrack);
    if (x->outL) outlet_free(x->outL);
    if (x->outR) outlet_free(x->outR);
}

// ---------- setup ----------
void juicy_bank_tilde_setup(void){
    juicy_bank_class = class_new(gensym("juicy_bank~"),
                                 (t_newmethod)juicy_bank_tilde_new,
                                 (t_method)juicy_bank_tilde_free,
                                 sizeof(t_juicy_bank_tilde),
                                 CLASS_DEFAULT,
                                 A_GIMME, 0);

    CLASS_MAINSIGNALIN(juicy_bank_class, t_juicy_bank_tilde, f_dummy);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);

    // messages
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_modes, gensym("modes"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_active, gensym("active"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_ratio_msg, gensym("ratio"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_decay_msg, gensym("decay"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_gain_msg,  gensym("gain"),  A_DEFFLOAT, A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_freqlist,  gensym("freq"),   A_GIMME, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_decaylist, gensym("decays"), A_GIMME, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_amplist,   gensym("amps"),   A_GIMME, 0);

    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_damping,    gensym("damping"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_position,   gensym("position"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_contact,    gensym("contact"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_contact_sym,gensym("contact_symmetry"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_density_msg,gensym("density"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);

    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_seed, gensym("seed"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_dispersion_reroll, gensym("dispersion_reroll"), 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_aniso_epsilon, gensym("aniso_epsilon"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_keytrack_on, gensym("keytrack_on"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_class, (t_method)juicy_bank_tilde_basef0,     gensym("basef0"), A_DEFFLOAT, 0);
}
