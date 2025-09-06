
// juicy_bank~ — modal resonator bank (fresh build w/ phase_random, contact oversampling, DC HPF)
// Compile (Linux):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//     -I"/usr/include/pd" -shared -fPIC -Wl,-export-dynamic -lm \
//     -o juicy_bank~.pd_linux juicy_bank_tilde.c
// Compile (macOS):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//     -I"/Applications/Pd-0.56-1.app/Contents/Resources/src" \
//     -arch arm64 -arch x86_64 -mmacosx-version-min=10.13 \
//     -bundle -undefined dynamic_lookup \
//     -o juicy_bank~.pd_darwin juicy_bank_tilde.c

#include "m_pd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>

static t_class *juicy_bank_tilde_class;

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#define JB_MAX_MODES 64

typedef struct {
    unsigned int s;
} jb_rng_t;

static inline void jb_rng_seed(jb_rng_t *r, unsigned int s){ if(!s) s=1; r->s = s; }
static inline unsigned int jb_rng_u32(jb_rng_t *r){
    // xorshift32
    unsigned int x = r->s;
    x ^= x << 13; x ^= x >> 17; x ^= x << 5;
    r->s = x; return x;
}
static inline float jb_rng_uni(jb_rng_t *r){ return (jb_rng_u32(r) >> 8) * (1.0f/16777216.0f); } // [0,1)
static inline float jb_rng_bi(jb_rng_t *r){ return 2.f * jb_rng_uni(r) - 1.f; }                  // [-1,1)

static inline float jb_clamp(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }
static inline float jb_lerp(float a, float b, float t){ return a + (b - a) * t; }

typedef enum { DENSITY_PIVOT=0, DENSITY_INDIV=1 } jb_density_mode;

typedef struct {
    // base params
    float base_ratio;     // if keytrack_on: ratio (1 = fundamental). else: absolute Hz
    float base_decay_ms;  // base T60 per mode
    float base_gain;      // 0..1
    float attack_ms;      // smoothing of drive
    float curve_amt;      // -1..1 envelope curve
    float pan;            // -1..1

    // runtime
    int   active;
    float ratio_now;
    float decay_ms_now;
    float gain_now;
    float a1,a2;
    float y1,y2;
    float drive;
    float env;

    // dispersion
    float disp_signature; // deterministic in [-1,1]

    // helpers
    int   hit_gate;
    int   hit_cool;
    float y_pre_last;
} jb_mode_t;

typedef struct _juicy_bank_tilde {
    t_object  x_obj;
    t_float   f_dummy;

    // signal
    t_float   sr;

    // bank
    int       n_modes;
    jb_mode_t m[JB_MAX_MODES];
    int       sel_index;     // 0-based
    float     sel_index_f;   // 1-based for inlet write

    // dispersion runtime
    float     disp_offset[JB_MAX_MODES];
    float     disp_target[JB_MAX_MODES];

    // global params
    float damping;     // 0..1 (percent cut of decay)
    float brightness;  // 0..1 (0.5 neutral)
    float position;    // 0..1 (0 bypass)
    float dispersion;  // 0..1 (±1 ratio max)
    float density_amt; // -1..1 (±50%)
    jb_density_mode density_mode;
    float aniso;       // -1..1
    float aniso_eps;   // near-integer epsilon
    float contact_amt; // 0..1
    float contact_sym; // -1..1
    float keytrack_amount; // 0..1
    int   keytrack_on; // 0/1
    float basef0;      // Hz

    // phase randomization
    float phase_rand;  // 0..1

    // RNG
    jb_rng_t rng;

    // DC highpass
    float hp_a;
    float hpL_x1, hpL_y1, hpR_x1, hpR_y1;

    // inlets (just to keep references if needed)
    t_inlet *inR;
    t_outlet *outL, *outR;
} t_juicy_bank_tilde;

// ---------- helpers

static int jb_is_near_integer(float x, float eps){
    float n = roundf(x);
    return fabsf(x - n) <= eps;
}

// env curve shape: -1 exp-ish, 0 linear, +1 log-ish
static float jb_shape_env(float e, float curve){
    e = jb_clamp(e, 0.f, 1.f);
    if (curve < 0.f){
        float p = 1.f + (-curve)*2.f; // 1..3
        return powf(e, p);
    } else if (curve > 0.f){
        float p = 1.f - 0.75f*curve;  // 1..0.25
        if (p < 0.05f) p = 0.05f;
        return 1.f - powf(1.f - e, p);
    } else {
        return e;
    }
}

// contact nonlinearity with symmetry + makeup, used at 2x oversampling points
static float jb_contact_shape(float x, float amt, float sym){
    // scale by amount and asymmetry, then tanh, then apply makeup to keep small-signal gain ≈ 1
    float a = 1.f + 2.f*jb_clamp(amt, 0.f, 1.f);
    float s = 1.f + 0.5f*jb_clamp(sym, -1.f, 1.f);
    float scale = (x >= 0.f) ? (a * s) : (a * (2.f - s)); // mirror
    float z = x * scale;
    float y = tanhf(z);
    float makeup = (scale > 1e-6f) ? (1.f/scale) : 1.f;
    return y * makeup;
}

// ---------- configuration / updates

static void jb_update_density(t_juicy_bank_tilde *x){
    float s = 1.f + 0.5f * jb_clamp(x->density_amt, -1.f, 1.f); // 0.5..1.5

    if (x->density_mode == DENSITY_PIVOT){
        // pivot = fundamental ratio ~ 1 (or first active's base)
        float r_pivot = 1.f;
        int fid = -1;
        // find a base ratio nearest to 1 among active modes
        float best = 1e9f;
        for(int i=0;i<x->n_modes;i++){
            if(!x->m[i].active) continue;
            float d = fabsf(x->m[i].base_ratio - 1.f);
            if (d < best){ best = d; fid = i; }
        }
        if (fid >= 0) r_pivot = x->m[fid].base_ratio;
        for(int i=0;i<x->n_modes;i++){
            jb_mode_t *md=&x->m[i];
            if(!md->active){ md->ratio_now = md->base_ratio; continue; }
            if (i == fid) md->ratio_now = md->base_ratio;
            else md->ratio_now = r_pivot + (md->base_ratio - r_pivot) * s;
        }
    } else {
        // individual: scale each gap vs previous ACTIVE mode
        int idxs[JB_MAX_MODES]; int count=0;
        for(int i=0;i<x->n_modes;i++) if(x->m[i].active) idxs[count++]=i;
        if(count==0){ for(int i=0;i<x->n_modes;i++) x->m[i].ratio_now=x->m[i].base_ratio; return; }
        // sort by base_ratio
        for(int k=1;k<count;k++){
            int id=idxs[k]; int j=k;
            while(j>0 && x->m[idxs[j-1]].base_ratio > x->m[id].base_ratio){ idxs[j]=idxs[j-1]; j--; }
            idxs[j]=id;
        }
        for(int j=0;j<count;j++){
            int i = idxs[j];
            jb_mode_t *md = &x->m[i];
            if (j==0){ md->ratio_now = md->base_ratio; }
            else {
                int prev = idxs[j-1];
                float gap = (x->m[i].base_ratio - x->m[prev].base_ratio) * s;
                md->ratio_now = x->m[prev].ratio_now + gap;
            }
        }
        // non-actives
        for(int i=0;i<x->n_modes;i++) if(!x->m[i].active) x->m[i].ratio_now=x->m[i].base_ratio;
    }
}

// brightness gain tilt around fundamental (0.5 neutral)
static float jb_bright_gain(float ratio_rel, float b){
    float t = (jb_clamp(b,0.f,1.f) - 0.5f) * 2.f; // -1..1
    // use mild power-law tilt: positive boosts highs, negative attenuates highs
    float p = 0.6f * t; // gentle
    if (p >= 0.f) return powf(jb_clamp(ratio_rel,1.f,1e6f), p);
    else          return powf(jb_clamp(ratio_rel,1.f,1e6f), p); // same expression, p<0 attenuates
}

// position weighting: 0 => bypass (1.0); >0 => |sin(pi * k * pos)|
static float jb_position_weight(float ratio_rel, float pos){
    if (pos <= 0.f) return 1.f;
    float k = roundf(jb_clamp(ratio_rel, 1.f, 1e6f));
    float w = fabsf(sinf((float)M_PI * k * jb_clamp(pos,0.f,1.f)));
    return w;
}

static void jb_update_per_mode_gains(t_juicy_bank_tilde *x){
    // compute gain_now from base_gain + global shapers
    for(int i=0;i<x->n_modes;i++){
        jb_mode_t *md=&x->m[i];
        if (!md->active){ md->gain_now=0.f; continue; }

        float ratio = md->ratio_now + x->disp_offset[i];
        if (ratio < 0.01f) ratio = 0.01f;

        // dispersion immune for fundamental: if near 1, ignore offset for ratio_rel calc
        float ratio_rel = ratio;
        // brightness tilt (0.5 neutral; lows not boosted)
        float g = md->base_gain * jb_bright_gain(ratio_rel, x->brightness);

        // anisotropy continuous-ish
        float a = x->aniso;
        float w = 1.f;
        int nearint = jb_is_near_integer(ratio, x->aniso_eps);
        if (a > 0.f){ // keep even (near-integers) progressively
            w = (nearint ? 1.f : (1.f - a));
        } else if (a < 0.f){ // keep odd progressively
            w = (!nearint ? 1.f : (1.f + a)); // a negative
        }
        if (w < 0.f) w = 0.f;

        // position
        float wp = jb_position_weight(ratio_rel, x->position);

        md->gain_now = g * w * wp;
        if (md->gain_now < 0.f) md->gain_now = 0.f;
    }
}

static void jb_update_coeffs(t_juicy_bank_tilde *x){
    // dispersion offsets (slew to targets)
    for(int i=0;i<x->n_modes;i++){
        float tgt = x->disp_target[i];
        float cur = x->disp_offset[i];
        float d = tgt - cur;
        x->disp_offset[i] = cur + 0.0025f * d; // gentle glide
    }

    // update density mapping
    jb_update_density(x);

    // compute coefficients per mode
    for(int i=0;i<x->n_modes;i++){
        jb_mode_t *md=&x->m[i];
        if (!md->active){ md->a1=md->a2=0.f; continue; }

        float ratio = md->ratio_now + x->disp_offset[i];
        // fundamental immune to dispersion offset for classification, but freq uses ratio (allow small offsets)
        if (i == 0) ratio = md->ratio_now; // assume mode 0 is fundamental by default

        float Hz = x->keytrack_on ? (x->basef0 * ratio) : ratio;
        Hz = jb_clamp(Hz, 0.f, 0.49f * x->sr);
        float w = 2.f * (float)M_PI * Hz / x->sr;

        md->decay_ms_now = md->base_decay_ms * (1.f - x->damping);
        float T60 = jb_clamp(md->decay_ms_now, 0.f, 1e7f) * 0.001f; // sec
        float r = (T60 <= 0.f) ? 0.f : powf(10.f, -3.f / (T60 * x->sr));

        // curve morph applied to radius
        float ca = md->curve_amt;
        float r_eff = r;
        if (ca < 0.f){ float p=1.f + (-ca)*2.f; r_eff = powf(r, p); }
        else if (ca > 0.f){ float p=1.f - 0.75f*ca; if (p<0.05f) p=0.05f; float one_m_r=1.f-r; if (one_m_r<1e-6f) one_m_r=1e-6f; r_eff = 1.f - powf(one_m_r, p); }

        float c = cosf(w);
        md->a1 = 2.f * r_eff * c;
        md->a2 = -r_eff * r_eff;
    }
}

// ---------- perform

static t_int *juicy_bank_tilde_perform(t_int *w){
    t_juicy_bank_tilde *x = (t_juicy_bank_tilde *)(w[1]);
    t_sample *inL = (t_sample *)(w[2]);
    t_sample *inR = (t_sample *)(w[3]);
    t_sample *outL= (t_sample *)(w[4]);
    t_sample *outR= (t_sample *)(w[5]);
    int n = (int)(w[6]);

    // zero outs
    for (int i=0;i<n;i++){ outL[i]=0; outR[i]=0; }

    // update coeffs & gains per block
    jb_update_coeffs(x);
    jb_update_per_mode_gains(x);

    // contact parameters (block constants)
    float camt = jb_clamp(x->contact_amt, 0.f, 1.f);
    float csym = jb_clamp(x->contact_sym, -1.f, 1.f);

    for (int m=0; m<x->n_modes; m++){
        jb_mode_t *md=&x->m[m];
        if (!md->active || md->gain_now <= 0.f) continue;

        float gl = sqrtf(0.5f * (1.f - md->pan));
        float gr = sqrtf(0.5f * (1.f + md->pan));

        float y1 = md->y1, y2 = md->y2, drive = md->drive, env = md->env;

        // drive smoothing coeff
        float att_ms = jb_clamp(md->attack_ms, 0.f, 500.f);
        float att_a = (att_ms <= 0.f) ? 1.f : (1.f - expf(-1.f / (0.001f * att_ms * x->sr)));

        float th = 1e-4f;

        for (int i=0;i<n;i++){
            float exc = (inL[i] + inR[i]);
            float target = md->gain_now * exc;
            float abs_target = fabsf(target);

            // per-hit phase randomization
            if (x->phase_rand > 0.f){
                if (md->hit_cool > 0) md->hit_cool--;
                else {
                    if (!md->hit_gate && abs_target > 1e-3f){
                        float k = x->phase_rand * 0.05f * abs_target;
                        float r1 = jb_rng_bi(&x->rng);
                        float r2 = jb_rng_bi(&x->rng);
                        y1 += k * r1;
                        y2 += k * r2;
                        md->hit_gate = 1;
                        md->hit_cool = (int)(x->sr * 0.005f); // 5 ms
                    }
                    if (abs_target < 5e-4f) md->hit_gate = 0;
                }
            }

            drive += att_a * (target - drive);

            // linear step
            float y_lin = (md->a1 * y1 + md->a2 * y2) + drive;
            y2 = y1; y1 = y_lin;

            float abs_y = fabsf(y_lin);
            env = env + 0.0015f * (abs_y - env); // simple env follower (block-independent)

            float y = y_lin;

            // contact nonlinearity @ 2x
            if (camt > 0.f && env > th){
                float mid = 0.5f * (md->y_pre_last + y_lin);
                float y_mid = jb_contact_shape(mid, camt, csym);
                float y_hi  = jb_contact_shape(y_lin, camt, csym);
                y = 0.5f * (y_mid + y_hi);
            }
            md->y_pre_last = y_lin;

            float e = (env > 1.f) ? 1.f : ((env < 0.f) ? 0.f : env);
            float e_sh = jb_shape_env(e, md->curve_amt);
            float yamp = y * e_sh;

            outL[i] += yamp * gl;
            outR[i] += yamp * gr;
        }

        md->y1 = y1; md->y2 = y2; md->drive = drive; md->env = env;
    }

    // DC high-pass on outputs (first-order at ~8 Hz)
    float a = x->hp_a;
    float x1L = x->hpL_x1, y1L = x->hpL_y1;
    float x1R = x->hpR_x1, y1R = x->hpR_y1;
    for (int i=0;i<n;i++){
        float xl = outL[i], xr = outR[i];
        float yl = a * (y1L + xl - x1L);
        float yr = a * (y1R + xr - x1R);
        // denormal guard
        if (fabsf(yl) < 1e-20f) yl = 0.f;
        if (fabsf(yr) < 1e-20f) yr = 0.f;
        outL[i] = yl; outR[i] = yr;
        x1L = xl; y1L = yl; x1R = xr; y1R = yr;
    }
    x->hpL_x1 = x1L; x->hpL_y1 = y1L; x->hpR_x1 = x1R; x->hpR_y1 = y1R;

    return (w + 7);
}

// ---------- messages

static void juicy_bank_tilde_active(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg onf){
    int idx = (int)idxf - 1; if (idx < 0 || idx >= x->n_modes) return;
    x->m[idx].active = (onf > 0.f) ? 1 : 0;
}

static void juicy_bank_tilde_modes(t_juicy_bank_tilde *x, t_floatarg nf){
    int n = (int)nf; if (n < 1) n = 1; if (n > JB_MAX_MODES) n = JB_MAX_MODES;
    x->n_modes = n;
}

static void juicy_bank_tilde_idx(t_juicy_bank_tilde *x, t_floatarg f){
    int idx = (int)f - 1;
    if (idx < 0) idx = 0;
    if (idx >= x->n_modes) idx = x->n_modes - 1;
    x->sel_index = idx; x->sel_index_f = f;
}

static void juicy_bank_tilde_ratio(t_juicy_bank_tilde *x, t_floatarg r){
    jb_mode_t *md=&x->m[x->sel_index];
    md->base_ratio = (r <= 0.f) ? 0.01f : r;
    md->ratio_now  = md->base_ratio;
}
static void juicy_bank_tilde_gain(t_juicy_bank_tilde *x, t_floatarg g){
    jb_mode_t *md=&x->m[x->sel_index];
    md->base_gain = jb_clamp(g, 0.f, 1.f);
}
static void juicy_bank_tilde_attack(t_juicy_bank_tilde *x, t_floatarg ms){
    jb_mode_t *md=&x->m[x->sel_index];
    md->attack_ms = (ms<0.f)?0.f:ms;
}
static void juicy_bank_tilde_decay(t_juicy_bank_tilde *x, t_floatarg ms){
    jb_mode_t *md=&x->m[x->sel_index];
    md->base_decay_ms = (ms<0.f)?0.f:ms;
}
static void juicy_bank_tilde_curve(t_juicy_bank_tilde *x, t_floatarg amt){
    jb_mode_t *md=&x->m[x->sel_index];
    if (amt < -1.f) amt = -1.f; if (amt > 1.f) amt = 1.f;
    md->curve_amt = amt;
}
static void juicy_bank_tilde_pan(t_juicy_bank_tilde *x, t_floatarg p){
    jb_mode_t *md=&x->m[x->sel_index];
    md->pan = jb_clamp(p, -1.f, 1.f);
}

static void juicy_bank_tilde_freq(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    for(int i=0;i<argc && i< x->n_modes; i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv + i);
            x->m[i].base_ratio = (v<=0.f)?0.01f:v;
            x->m[i].ratio_now  = x->m[i].base_ratio;
        }
    }
}
static void juicy_bank_tilde_decays(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    for(int i=0;i<argc && i< x->n_modes; i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv + i);
            x->m[i].base_decay_ms = (v<0.f)?0.f:v;
        }
    }
}
static void juicy_bank_tilde_amps(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    for(int i=0;i<argc && i< x->n_modes; i++){
        if (argv[i].a_type == A_FLOAT){
            float v = atom_getfloat(argv + i);
            x->m[i].base_gain = jb_clamp(v, 0.f, 1.f);
        }
    }
}

static void juicy_bank_tilde_damping(t_juicy_bank_tilde *x, t_floatarg f){ x->damping = jb_clamp(f, 0.f, 1.f); }
static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness = jb_clamp(f, 0.f, 1.f); }
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){ x->position = (f<=0.f)?0.f:jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){
    x->dispersion = jb_clamp(f, 0.f, 1.f);
    // retarget
    for(int i=0;i<x->n_modes;i++){
        jb_mode_t *md=&x->m[i];
        if (!md->active){ x->disp_target[i] = 0.f; continue; }
        // fundamental immune
        if (i==0){ x->disp_target[i] = 0.f; continue; }
        x->disp_target[i] = jb_clamp(md->disp_signature * x->dispersion, -1.f, 1.f);
    }
}
static void juicy_bank_tilde_seed(t_juicy_bank_tilde *x, t_floatarg f){
    jb_rng_seed(&x->rng, (unsigned int)((int)f*2654435761u));
    for(int i=0;i<x->n_modes;i++){
        x->m[i].disp_signature = jb_rng_bi(&x->rng);
    }
}
static void juicy_bank_tilde_dispersion_reroll(t_juicy_bank_tilde *x){
    for(int i=0;i<x->n_modes;i++){
        x->m[i].disp_signature = jb_rng_bi(&x->rng);
    }
    // refresh targets
    juicy_bank_tilde_dispersion(x, x->dispersion);
}

static void juicy_bank_tilde_density(t_juicy_bank_tilde *x, t_floatarg f){ x->density_amt = jb_clamp(f, -1.f, 1.f); }
static void juicy_bank_tilde_density_pivot(t_juicy_bank_tilde *x){ x->density_mode = DENSITY_PIVOT; }
static void juicy_bank_tilde_density_individual(t_juicy_bank_tilde *x){ x->density_mode = DENSITY_INDIV; }

static void juicy_bank_tilde_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso = jb_clamp(f, -1.f, 1.f); }
static void juicy_bank_tilde_aniso_eps(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso_eps = jb_clamp(f, 0.f, 0.25f); }

static void juicy_bank_tilde_contact(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_amt = jb_clamp(f, 0.f, 1.f); }
static void juicy_bank_tilde_contact_sym(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_sym = jb_clamp(f, -1.f, 1.f); }

static void juicy_bank_tilde_phase_random(t_juicy_bank_tilde *x, t_floatarg f){ x->phase_rand = jb_clamp(f, 0.f, 1.f); }

static void juicy_bank_tilde_basef0(t_juicy_bank_tilde *x, t_floatarg f){ x->basef0 = (f<=0.f)?1.f:f; }
static void juicy_bank_tilde_base_alias(t_juicy_bank_tilde *x, t_floatarg f){ juicy_bank_tilde_basef0(x,f); }
static void juicy_bank_tilde_keytrack_on(t_juicy_bank_tilde *x, t_floatarg f){ x->keytrack_on = (f>0.f)?1:0; }

static void juicy_bank_tilde_reset(t_juicy_bank_tilde *x){
    // Single-mode test: mode 1 active, ratio=1, gain=0.5, decay=500ms
    for(int i=0;i<x->n_modes;i++){
        x->m[i].active=0; x->disp_offset[i]=0.f; x->disp_target[i]=0.f;
        x->m[i].y1 = x->m[i].y2 = x->m[i].drive = x->m[i].env = 0.f;
        x->m[i].hit_gate=0; x->m[i].hit_cool=0; x->m[i].y_pre_last=0.f;
    }
    x->n_modes = 1;
    x->sel_index = 0; x->sel_index_f = 1.f;
    jb_mode_t *md = &x->m[0];
    md->active = 1;
    md->base_ratio = 1.f; md->ratio_now=1.f;
    md->base_gain  = 0.5f; md->gain_now=0.5f;
    md->base_decay_ms = 500.f; md->decay_ms_now = md->base_decay_ms * (1.f - x->damping);
    md->curve_amt = 0.f; md->attack_ms=0.f; md->pan=0.f;
}
static void juicy_bank_tilde_restart(t_juicy_bank_tilde *x){ juicy_bank_tilde_reset(x); }

// ---------- new/free/dsp/setup

static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr;
    float fc = 8.f; float RC = 1.f/(2.f*M_PI*fc); float dt = 1.f/x->sr; x->hp_a = RC/(RC+dt);
    dsp_add(juicy_bank_tilde_perform, 6, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[3]->s_vec, sp[0]->s_n);
}

static void juicy_bank_tilde_free(t_juicy_bank_tilde *x){
    inlet_free(x->inR);
    outlet_free(x->outL);
    outlet_free(x->outR);
}

static void *juicy_bank_tilde_new(void){
    t_juicy_bank_tilde *x = (t_juicy_bank_tilde *)pd_new(juicy_bank_tilde_class);
    // Default config
    x->sr = sys_getsr();
    if (x->sr <= 0) x->sr = 48000;
    x->n_modes = 20;        // << default 20
    x->sel_index = 0; x->sel_index_f = 1.f;

    x->damping=0.f; x->brightness=0.5f; x->position=0.f;
    x->dispersion=0.f; x->density_amt=0.f; x->density_mode=DENSITY_PIVOT;
    x->aniso=0.f; x->aniso_eps=0.02f;
    x->contact_amt=0.f; x->contact_sym=0.f;
    x->keytrack_on=1; x->keytrack_amount=1.f; x->basef0=440.f;
    x->phase_rand = 0.4f;
    jb_rng_seed(&x->rng, 0xC0FFEEu);

    x->hp_a=0.f; x->hpL_x1=x->hpL_y1=x->hpR_x1=x->hpR_y1=0.f;

    for(int i=0;i<JB_MAX_MODES;i++){ x->disp_offset[i]=0.f; x->disp_target[i]=0.f; }

    for(int i=0;i<JB_MAX_MODES;i++){
        jb_mode_t *md=&x->m[i];
        md->active = (i < 20);
        md->base_ratio = (float)(i+1);
        md->ratio_now  = md->base_ratio;
        md->base_decay_ms = 500.f;
        md->decay_ms_now  = md->base_decay_ms;
        md->base_gain = 0.2f;
        md->gain_now  = md->base_gain;
        md->attack_ms = 0.f;
        md->curve_amt = 0.f;
        if (i==0) md->pan = 0.f; else md->pan = ((i & 1) ? -0.2f : 0.2f);
        md->a1=md->a2=md->y1=md->y2=md->drive=0.f;
        md->env=0.f;
        md->disp_signature = 0.f;
        md->hit_gate=0; md->hit_cool=0; md->y_pre_last=0.f;
    }

    x->inR = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);

    return (void *)x;
}

void juicy_bank_tilde_setup(void){
    juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
                           (t_newmethod)juicy_bank_tilde_new,
                           (t_method)juicy_bank_tilde_free,
                           sizeof(t_juicy_bank_tilde),
                           CLASS_DEFAULT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);
    CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy);

    // structure
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_modes, gensym("modes"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_active, gensym("active"), A_DEFFLOAT, A_DEFFLOAT, 0);

    // select + per-mode setters
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_idx, gensym("idx"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_ratio, gensym("ratio"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_gain,  gensym("gain"),  A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_attack,gensym("attack"),A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay, gensym("decay"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_curve, gensym("curve"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pan,   gensym("pan"),   A_DEFFLOAT, 0);

    // batch
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_freq,   gensym("freq"),   A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decays, gensym("decays"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_amps,   gensym("amps"),   A_GIMME, 0);

    // globals
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damping,    gensym("damping"),    A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position,   gensym("position"),   A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_seed,       gensym("seed"),       A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion_reroll, gensym("dispersion_reroll"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density,    gensym("density"),    A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_aniso_eps,  gensym("aniso_epsilon"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact,    gensym("contact"),    A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact_sym,gensym("contact_symmetry"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_random, gensym("phase_random"), A_DEFFLOAT, 0);

    // pitch
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_basef0,   gensym("basef0"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_base_alias, gensym("base"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_keytrack_on, gensym("keytrack_on"), A_DEFFLOAT, 0);

    // utility
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_reset,   gensym("reset"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_restart, gensym("restart"), 0);

    class_sethelpsymbol(juicy_bank_tilde_class, gensym("juicy_bank~"));
}
