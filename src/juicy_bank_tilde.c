// juicy_bank~ — modal resonator bank (V4.9)
// 4-voice poly, true stereo banks, Behavior + Body + Individual inlets.
// NEW (V4.9):
//   • **Per-voice exciter inputs**: 8 extra signal inlets [v1_L v1_R v2_L v2_R v3_L v3_R v4_L v4_R]
//     Use message **exciter_mode 1** to drive each voice from its own stereo input (pair comes from juicy_exciter~).
//     Default **exciter_mode 0** keeps legacy 2-in global excitation (stereo) feeding held voices.
//   • Backward compatible: if you don't send exciter_mode 1, patch behaves exactly like older builds.
//
// INLET GROUPS (left → right):
//  • SIGNAL (10 total): inL, inR, v1_L, v1_R, v2_L, v2_R, v3_L, v3_R, v4_L, v4_R
//  • BEHAVIOR (7): stiffen, shortscle, linger, tilt, bite, bloom, crossring
//  • BODY (7):     damping, brightness, position, density, dispersion, anisotropy, contact
//  • INDIVIDUAL (8, per-mode via index): index, ratio, gain, attack, decya, curve, pan, keytrack
//
// Voice-addressed poly for Pd [poly]:
//  • note_poly <v> <Hz> <vel>  • note_poly_midi <v> <midinote> <vel>  • off_poly <v>
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

#define JB_MAX_MODES   64
#define JB_MAX_VOICES   4

// ---------- utils ----------
static inline float jb_clamp(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }
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
    // runtime per-mode
    float ratio_now, decay_ms_now, gain_now;
    float t60_s, decay_u;

    // per-ear per-hit randomizations
    float md_hit_offsetL, md_hit_offsetR;   // micro detune offsets
    float bw_hit_ratioL, bw_hit_ratioR;     // twin detune ratios

    // LEFT states
    float a1L,a2L, y1L,y2L, a1bL,a2bL, y1bL,y2bL, envL, y_pre_lastL;
    // RIGHT states
    float a1R,a2R, y1R,y2R, a1bR,a2bR, y1bR,y2bR, envR, y_pre_lastR;

    // drive/hit
    float driveL, driveR;
    int   hit_gateL, hit_coolL, hit_gateR, hit_coolR;
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

    // runtime per-mode
    jb_mode_rt_t m[JB_MAX_MODES];
} jb_voice_t;

// ---------- the object ----------
static t_class *juicy_bank_tilde_class;

typedef struct _juicy_bank_tilde {
    t_object  x_obj; t_float f_dummy; t_float sr;

    int n_modes;
    jb_mode_base_t base[JB_MAX_MODES];

    // BODY globals
    float damping, brightness, position;
    float density_amt; jb_density_mode density_mode;
    float dispersion, dispersion_last;
    float aniso, aniso_eps;
    float contact_amt, contact_sym;

    // realism/misc
    float phase_rand; int phase_debug;
    float bandwidth;        // base for Bloom
    float micro_detune;     // base for micro detune
    float basef0_ref;

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
    // Body
    t_inlet *in_damping, *in_brightness, *in_position, *in_density, *in_dispersion, *in_aniso, *in_contact;
    // Individual
    t_inlet *in_index, *in_ratio, *in_gain, *in_attack, *in_decya, *in_curve, *in_pan, *in_keytrack;
} t_juicy_bank_tilde;


static void juicy_bank_tilde_reset(t_juicy_bank_tilde *x);
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
    float s = 1.f + 0.5f * jb_clamp(x->density_amt, -1.f, 1.f);
    int idxs[JB_MAX_MODES], count=0;
    for(int i=0;i<x->n_modes;i++){
        if(x->base[i].active && x->base[i].keytrack) idxs[count++]=i;
        else v->m[i].ratio_now = x->base[i].base_ratio;
    }
    if(count==0) return;

    // insertion sort by base_ratio
    for(int k=1;k<count;k++){
        int id=idxs[k], j=k;
        while(j>0 && x->base[idxs[j-1]].base_ratio > x->base[id].base_ratio){ idxs[j]=idxs[j-1]; j--; }
        idxs[j]=id;
    }

    if (x->density_mode == DENSITY_PIVOT){
        int fid=-1; float best=1e9f;
        for(int i=0;i<count;i++){
            int id=idxs[i]; float d=fabsf(x->base[id].base_ratio-1.f); if(d<best){best=d; fid=id;}
        }
        float r_pivot = (fid>=0) ? x->base[fid].base_ratio : 1.f;
        for(int i=0;i<count;i++){
            int m=idxs[i];
            if(m==fid) v->m[m].ratio_now = x->base[m].base_ratio;
            else v->m[m].ratio_now = r_pivot + (x->base[m].base_ratio - r_pivot) * s;
        }
    } else {
        for(int j=0;j<count;j++){
            int i=idxs[j];
            if(j==0) v->m[i].ratio_now=x->base[i].base_ratio;
            else {
                int prev=idxs[j-1];
                float gap=(x->base[i].base_ratio-x->base[prev].base_ratio)*s;
                v->m[i].ratio_now=v->m[prev].ratio_now+gap;
            }
        }
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

    // Tilt + Bite → brightness
    float dbright_pitch = (0.02f + 0.08f * x->tilt_amt) * log2f(xfac);
    float dbright_vel   = (0.25f + 0.35f * x->bite_amt) * jb_clamp(v->vel,0.f,1.f);
    v->brightness_v = jb_clamp(x->brightness + dbright_pitch + dbright_vel, 0.f, 1.f);

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

// ---------- sympathetic update ----------
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
static void jb_update_voice_coeffs(t_juicy_bank_tilde *x, jb_voice_t *v){
    for(int i=0;i<x->n_modes;i++){
        float d=v->disp_target[i]-v->disp_offset[i];
        v->disp_offset[i]+=0.0025f*d;
    }

    jb_apply_density(x, v);

    float md_amt = jb_clamp(x->micro_detune,0.f,1.f);
    float bw_amt = jb_clamp(v->bandwidth_v, 0.f, 1.f);

    for(int i=0;i<x->n_modes;i++){
        jb_mode_rt_t *md=&v->m[i];
        if(!x->base[i].active){
            md->a1L=md->a2L=md->a1bL=md->a2bL=0.f;
            md->a1R=md->a2R=md->a1bR=md->a2bR=0.f;
            md->t60_s=0.f;
            continue;
        }

        float ratio_base = md->ratio_now + v->disp_offset[i];
        float ratioL = ratio_base;
        float ratioR = ratio_base;
        if(i!=0){ ratioL += md_amt * md->md_hit_offsetL; ratioR += md_amt * md->md_hit_offsetR; }
        if (ratioL < 0.01f) ratioL = 0.01f; if (ratioR < 0.01f) ratioR = 0.01f;

        float HzL = x->base[i].keytrack ? (v->f0 * ratioL) : ratioL;
        float HzR = x->base[i].keytrack ? (v->f0 * ratioR) : ratioR;
        HzL = jb_clamp(HzL, 0.f, 0.49f*x->sr);
        HzR = jb_clamp(HzR, 0.f, 0.49f*x->sr);
        float wL = 2.f * (float)M_PI * HzL / x->sr;
        float wR = 2.f * (float)M_PI * HzR / x->sr;

        float base_ms = x->base[i].base_decay_ms;
        float T60 = jb_clamp(base_ms, 0.f, 1e7f) * 0.001f;
        T60 *= (1.f - 0.5f * jb_clamp(x->damping, -1.f, 1.f));
        T60 *= v->decay_pitch_mul;
        T60 *= v->decay_vel_mul;
        T60 *= v->cr_decay_mul[i];
        md->t60_s = T60;

        float r = (T60 <= 0.f) ? 0.f : powf(10.f, -3.f / (T60 * x->sr));
        float cL=cosf(wL), cR=cosf(wR);

        md->a1L=2.f*r*cL; md->a2L=-r*r;
        md->a1R=2.f*r*cR; md->a2R=-r*r;

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
        float ratio_rel = x->base[i].keytrack ? ratio : ((v->f0>0.f)? (ratio / v->f0) : ratio);

        float g = x->base[i].base_gain * jb_bright_gain(ratio_rel, v->brightness_v);

        float a = x->aniso; float w = 1.f;
        int nearint = jb_is_near_integer(ratio_rel, x->aniso_eps);
        if (a > 0.f){ w = (nearint ? 1.f : (1.f - a)); }
        else if (a < 0.f){ w = (!nearint ? 1.f : (1.f + a)); }
        if(w<0.f) w=0.f;

        float wp = jb_position_weight(ratio_rel, x->position);

        g *= v->cr_gain_mul[i];

        float gn = g * w * wp;
        v->m[i].gain_now = (gn<0.f)?0.f:gn;
    }
}

// ---------- allocator helpers ----------
static void jb_voice_reset_states(const t_juicy_bank_tilde *x, jb_voice_t *v, jb_rng_t *rng){
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

    for(int i=0;i<n;i++){ outL[i]=0; outR[i]=0; }

    // block updates
    for(int vix=0; vix<x->max_voices; ++vix){
        jb_voice_t *v = &x->v[vix];
        if (v->state==V_IDLE) continue;
        jb_update_crossring(x, vix);
        jb_update_voice_coeffs(x, v);
        jb_update_voice_gains(x, v);
    }

    float camt=jb_clamp(x->contact_amt,0.f,1.f);
    float csym=jb_clamp(x->contact_sym,-1.f,1.f);
    int phase_hits_block=0;

    // choose buffer arrays for per-voice mode
    t_sample *vinL[JB_MAX_VOICES] = { v1L, v2L, v3L, v4L };
    t_sample *vinR[JB_MAX_VOICES] = { v1R, v2R, v3R, v4R };

    for(int vix=0; vix<x->max_voices; ++vix){
        jb_voice_t *v = &x->v[vix];
        if (v->state==V_IDLE) continue;

        float bw_amt = jb_clamp(v->bandwidth_v, 0.f, 1.f);
        float twin_mix = 0.12f * bw_amt;
        // excitation gating:
        //  • legacy mode (0): only HELD voices listen to the shared inL/inR
        //  • per-voice mode (1): the dedicated input carries its own envelope, so no gating
        const float use_gate = (x->exciter_mode==0) ? ((v->state==V_HELD)?1.f:0.f) : 1.f;
        t_sample *srcL = (x->exciter_mode==0) ? inL : vinL[vix];
        t_sample *srcR = (x->exciter_mode==0) ? inR : vinR[vix];

        for(int m=0;m<x->n_modes;m++){
            if(!x->base[m].active || v->m[m].gain_now<=0.f) continue;
            jb_mode_rt_t *md=&v->m[m];

            float y1L=md->y1L, y2L=md->y2L, y1bL=md->y1bL, y2bL=md->y2bL, driveL=md->driveL, envL=md->envL;
            float y1R=md->y1R, y2R=md->y2R, y1bR=md->y1bR, y2bR=md->y2bR, driveR=md->driveR, envR=md->envR;
            float u = md->decay_u;
            float att_ms = jb_clamp(x->base[m].attack_ms,0.f,500.f);
            float att_a = (att_ms<=0.f)?1.f:(1.f-expf(-1.f/(0.001f*att_ms*x->sr)));
            float th = 1e-4f;
            float du = (md->t60_s > 1e-6f) ? (1.f / (md->t60_s * x->sr)) : 1.f;

            for(int i=0;i<n;i++){
                // LEFT
                float excL = use_gate * srcL[i] * md->gain_now;
                float absL = fabsf(excL);
                if(absL>1e-3f){
                    if(md->hit_coolL>0){ md->hit_coolL--; }
                    if(!md->hit_gateL){
                        if(x->phase_rand>0.f){
                            float k=x->phase_rand*0.05f*absL;
                            float r1=jb_rng_bi(&x->rng), r2=jb_rng_bi(&x->rng);
                            y1L+=k*r1; y2L+=k*r2;
                            if (bw_amt>0.f){
                                float r3=jb_rng_bi(&x->rng), r4=jb_rng_bi(&x->rng);
                                y1bL+=k*r3; y2bL+=k*r4;
                            }
                            phase_hits_block++;
                        }
                        if(m!=0){ md->md_hit_offsetL = 0.05f * jb_rng_bi(&x->rng); } else { md->md_hit_offsetL = 0.f; }
                        {
                            float mode_scale = (x->n_modes>1)? ((float)m/(float)(x->n_modes-1)) : 0.f;
                            float max_det = 0.0005f + 0.0015f * mode_scale;
                            md->bw_hit_ratioL = max_det * jb_rng_bi(&x->rng);
                        }
                        md->hit_gateL=1; md->hit_coolL=(int)(x->sr*0.005f);
                        u=0.f;
                    }
                } else {
                    md->hit_gateL=0;
                }

                driveL += att_a*(excL-driveL);
                float y_linL = (md->a1L*y1L + md->a2L*y2L) + driveL; y2L=y1L; y1L=y_linL;

                float y_totalL = y_linL;
                if (bw_amt > 0.f){
                    float y_lin_bL = (md->a1bL*y1bL + md->a2bL*y2bL); y2bL=y1bL; y1bL=y_lin_bL;
                    y_totalL += 0.12f * bw_amt * y_lin_bL;
                }

                // RIGHT
                float excR = use_gate * srcR[i] * md->gain_now;
                float absR = fabsf(excR);
                if(absR>1e-3f){
                    if(md->hit_coolR>0){ md->hit_coolR--; }
                    if(!md->hit_gateR){
                        if(x->phase_rand>0.f){
                            float k=x->phase_rand*0.05f*absR;
                            float r1=jb_rng_bi(&x->rng), r2=jb_rng_bi(&x->rng);
                            y1R+=k*r1; y2R+=k*r2;
                            if (bw_amt>0.f){
                                float r3=jb_rng_bi(&x->rng), r4=jb_rng_bi(&x->rng);
                                y1bR+=k*r3; y2bR+=k*r4;
                            }
                            phase_hits_block++;
                        }
                        if(m!=0){ md->md_hit_offsetR = 0.05f * jb_rng_bi(&x->rng); } else { md->md_hit_offsetR = 0.f; }
                        {
                            float mode_scale = (x->n_modes>1)? ((float)m/(float)(x->n_modes-1)) : 0.f;
                            float max_det = 0.0005f + 0.0015f * mode_scale;
                            md->bw_hit_ratioR = max_det * jb_rng_bi(&x->rng);
                        }
                        md->hit_gateR=1; md->hit_coolR=(int)(x->sr*0.005f);
                        u=0.f;
                    }
                } else {
                    md->hit_gateR=0;
                }

                driveR += att_a*(excR-driveR);
                float y_linR = (md->a1R*y1R + md->a2R*y2R) + driveR; y2R=y1R; y1R=y_linR;

                float y_totalR = y_linR;
                if (bw_amt > 0.f){
                    float y_lin_bR = (md->a1bR*y1bR + md->a2bR*y2bR); y2bR=y1bR; y1bR=y_lin_bR;
                    y_totalR += 0.12f * bw_amt * y_lin_bR;
                }

                // curve shaping
                float S = jb_curve_shape_gain(u, x->base[m].curve_amt);
                y_totalL *= S; y_totalR *= S;
                u += du; if(u>1.f){ u=1.f; }

                // contact nonlinearity
                if(camt>0.f){
                    if (envL > th){
                        float mid=0.5f*(md->y_pre_lastL + y_totalL);
                        float y_mid= tanhf(mid * (1.f + 2.f*camt*(1.f+0.5f*jb_clamp(csym,-1.f,1.f))));
                        float y_hi = tanhf(y_totalL * (1.f + 2.f*camt*(1.f+0.5f*jb_clamp(csym,-1.f,1.f))));
                        y_totalL = 0.5f*(y_mid+y_hi);
                    }
                    if (envR > th){
                        float mid=0.5f*(md->y_pre_lastR + y_totalR);
                        float y_mid= tanhf(mid * (1.f + 2.f*camt*(1.f+0.5f*jb_clamp(csym,-1.f,1.f))));
                        float y_hi = tanhf(y_totalR * (1.f + 2.f*camt*(1.f+0.5f*jb_clamp(csym,-1.f,1.f))));
                        y_totalR = 0.5f*(y_mid+y_hi);
                    }
                }

                // output sum
                outL[i] += y_totalL;
                outR[i] += y_totalR;

                // update envelopes
                float ayL=fabsf(y_totalL); envL = envL + 0.0015f*(ayL - envL); md->y_pre_lastL = y_totalL;
                float ayR=fabsf(y_totalR); envR = envR + 0.0015f*(ayR - envR); md->y_pre_lastR = y_totalR;
            }

            md->y1L=y1L; md->y2L=y2L; md->y1bL=y1bL; md->y2bL=y2bL; md->driveL=driveL; md->envL=envL;
            md->y1R=y1R; md->y2R=y2R; md->y1bR=y1bR; md->y2bR=y2bR; md->driveR=driveR; md->envR=envR;
            md->decay_u=u;
        }

        float lastL = outL[n-1], lastR = outR[n-1];
        float e = 0.997f*v->energy + 0.003f*(fabsf(lastL)+fabsf(lastR));
        v->energy = e;
        if (v->state==V_RELEASE && e < 1e-6f){ v->state = V_IDLE; }
    }

    // DC high-pass
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
    int i=x->edit_idx; if(i<0||i>=x->n_modes) return; float v=(r<=0.f)?0.01f:r; x->base[i].base_ratio=v;
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
static void juicy_bank_tilde_damping(t_juicy_bank_tilde *x, t_floatarg f){ x->damping=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){ x->position=(f<=0.f)?0.f:jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_density(t_juicy_bank_tilde *x, t_floatarg f){ x->density_amt=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_density_pivot(t_juicy_bank_tilde *x){ x->density_mode=DENSITY_PIVOT; }
static void juicy_bank_tilde_density_individual(t_juicy_bank_tilde *x){ x->density_mode=DENSITY_INDIV; }
static void juicy_bank_tilde_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_aniso_eps(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso_eps=jb_clamp(f,0.f,0.25f); }
static void juicy_bank_tilde_contact(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_contact_sym(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_sym=jb_clamp(f,-1.f,1.f); }

// realism & misc
static void juicy_bank_tilde_phase_random(t_juicy_bank_tilde *x, t_floatarg f){ x->phase_rand=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_phase_debug(t_juicy_bank_tilde *x, t_floatarg on){ x->phase_debug=(on>0.f)?1:0; }
static void juicy_bank_tilde_bandwidth(t_juicy_bank_tilde *x, t_floatarg f){ x->bandwidth=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_micro_detune(t_juicy_bank_tilde *x, t_floatarg f){ x->micro_detune=jb_clamp(f,0.f,1.f); }

// dispersion & seeds
static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){
    float v=jb_clamp(f,0.f,1.f);
    if(x->dispersion_last<0.f || fabsf(v-x->dispersion_last)>1e-6f){
        for(int i=0;i<x->n_modes;i++){ x->base[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng); }
        x->dispersion_last=v;
    }
    x->dispersion=v;
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
static void juicy_bank_tilde_note_midi(t_juicy_bank_tilde *x, t_floatarg midi, t_floatarg vel){
    jb_note_on(x, jb_midi_to_hz(midi), vel);
}
static void juicy_bank_tilde_off(t_juicy_bank_tilde *x, t_floatarg f0){
    jb_note_off(x, (f0<=0.f)?1.f:f0);
}
static void juicy_bank_tilde_voices(t_juicy_bank_tilde *x, t_floatarg nf){
    (void)nf; x->max_voices = JB_MAX_VOICES; // fixed 4
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
        x->v[v].state = V_IDLE; x->v[v].f0 = x->basef0_ref; x->v[v].vel = 0.f; x->v[v].energy=0.f;
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
    float fc=8.f; float RC=1.f/(2.f*M_PI*fc); float dt=1.f/x->sr; x->hp_a=RC/(RC+dt);

    // sp layout: [inL, inR, v1L, v1R, v2L, v2R, v3L, v3R, v4L, v4R, outL, outR]
    t_int argv[2 + 12 + 1];
    int a=0;
    argv[a++] = (t_int)x;
    for(int k=0;k<12;k++) argv[a++] = (t_int)(sp[k]->s_vec);
    argv[a++] = (int)(sp[0]->s_n);
    dsp_addv(juicy_bank_tilde_perform, a, argv);
}

static void juicy_bank_tilde_free(t_juicy_bank_tilde *x){inlet_free(x->in_crossring);

    inlet_free(x->in_damping); inlet_free(x->in_brightness); inlet_free(x->in_position);
    inlet_free(x->in_density); inlet_free(x->in_dispersion); inlet_free(x->in_aniso); inlet_free(x->in_contact);

    inlet_free(x->in_index); inlet_free(x->in_ratio); inlet_free(x->in_gain);
    inlet_free(x->in_attack); inlet_free(x->in_decya); inlet_free(x->in_curve); inlet_free(x->in_pan); inlet_free(x->in_keytrack);

    inlet_free(x->inR);
    for(int i=0;i<JB_MAX_VOICES;i++){ if(x->in_vL[i]) inlet_free(x->in_vL[i]); if(x->in_vR[i]) inlet_free(x->in_vR[i]); }

    outlet_free(x->outL); outlet_free(x->outR);
}

// ---------- new() ----------
static void *juicy_bank_tilde_new(void){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)pd_new(juicy_bank_tilde_class);
    x->sr = sys_getsr(); if(x->sr<=0) x->sr=48000;

    x->n_modes=20; x->edit_idx=0;
    for(int i=0;i<JB_MAX_MODES;i++){
        x->base[i].active=(i<20);
        x->base[i].base_ratio=(float)(i+1);
        x->base[i].base_decay_ms=500.f;
        x->base[i].base_gain=0.2f;
        x->base[i].attack_ms=0.f;
        x->base[i].curve_amt=0.f;
        x->base[i].pan=(i==0)?0.f:((i&1)?-0.2f:0.2f);
        x->base[i].keytrack=1;
        x->base[i].disp_signature = 0.f;
        x->base[i].micro_sig      = 0.f;
    }

    // body defaults
    x->damping=0.f; x->brightness=0.5f; x->position=0.f;
    x->density_amt=0.f; x->density_mode=DENSITY_PIVOT;
    x->dispersion=0.f; x->dispersion_last=-1.f;
    x->aniso=0.f; x->aniso_eps=0.02f;
    x->contact_amt=0.f; x->contact_sym=0.f;

    // realism defaults
    x->phase_rand=1.f; x->phase_debug=0;
    x->bandwidth=0.1f; x->micro_detune=0.1f;

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

    // Behavior (reduced)    x->in_shortscle  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("shortscle")); // alias of shortscale    x->in_crossring  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("crossring"));

    // Body
    x->in_damping    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damping"));
    x->in_brightness = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("brightness"));
    x->in_position   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("position"));
    x->in_density    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("density"));
    x->in_dispersion = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("dispersion"));
    x->in_aniso      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("anisotropy"));
    x->in_contact    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("contact"));

    // Individual
    x->in_index      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("index"));
    x->in_ratio      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("ratio"));
    x->in_gain       = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("gain"));
    x->in_attack     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("attack"));
    x->in_decya      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("decya"));   // alias of decay
    x->in_curve      = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("curve"));
    x->in_pan        = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("pan"));
    x->in_keytrack   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("keytrack"));

    // Outs
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);
    return (void *)x;
}

// ---------- setup ----------

// INIT: scratch mode (only mode 1 active with default center values)
static void juicy_bank_tilde_INIT(t_juicy_bank_tilde *x){
    int n = x->n_modes; if (n < 1) n = 1;
    for (int i = 0; i < n; ++i){
        if (i == 0){
            x->base[i].active = 1;
            x->base[i].base_ratio = 1.f;
            x->base[i].base_gain = 0.5f;
            x->base[i].curve_amt = 0.f;       // linear
            x->base[i].base_decay_ms = 500.f; // "middle"
            x->base[i].attack_ms = 0.f;
            x->base[i].pan = 0.f;
            x->base[i].keytrack = 1;
        } else {
            x->base[i].active = 1;   // leave modes activatable
            x->base[i].base_gain = 0.f;
        }
    }
    x->edit_idx = 0;
    juicy_bank_tilde_reset(x);
} else {
            x->base[i].active = 0;
            x->base[i].base_gain = 0.f;
        }
    }
    x->edit_idx = 0;
    juicy_bank_tilde_reset(x);
}

void juicy_bank_tilde_setup(void){
    juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
                           (t_newmethod)juicy_bank_tilde_new,
                           (t_method)juicy_bank_tilde_free,
                           sizeof(t_juicy_bank_tilde), CLASS_DEFAULT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);
    CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy);

    // BEHAVIOR
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_stiffen, gensym("stiffen"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_shortscale, gensym("shortscale"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_shortscale, gensym("shortscle"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_linger, gensym("linger"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_tilt, gensym("tilt"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bite, gensym("bite"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bloom, gensym("bloom"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_crossring, gensym("crossring"), A_DEFFLOAT, 0);

    // BODY
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damping, gensym("damping"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position, gensym("position"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density, gensym("density"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_aniso_eps, gensym("aniso_epsilon"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact, gensym("contact"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact_sym, gensym("contact_symmetry"), A_DEFFLOAT, 0);

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
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
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

    // exciter mode
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_exciter_mode, gensym("exciter_mode"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_reset, gensym("reset"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_restart, gensym("restart"), 0);

    
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_INIT, gensym("INIT"), 0);
class_sethelpsymbol(juicy_bank_tilde_class, gensym("juicy_bank~"));
}
