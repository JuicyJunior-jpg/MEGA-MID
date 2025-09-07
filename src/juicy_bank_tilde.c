// juicy_bank~ — modal resonator bank (V3: proper curve shaping + bandwidth/micro_detune defaults + deeper micro_detune)
// Changes in this version:
//  1) Curve shaping is now a TRUE time-warp envelope that adapts to any decay length (ms).
//     - curve = -1 .. +1 : -1 = full exponential (fast early, slow later),
//                           0  = linear,
//                           +1 = full logarithmic (slow early, fast later).
//     - Implemented by multiplying the resonator output by S(u) = 10^{-3*(phi(u)-u)},
//       where u = t/T60 (0..1), and phi(u) = u^gamma with gamma mapped from curve:
//         curve < 0: gamma ∈ [0.35..1]  (more convex, faster early)
//         curve > 0: gamma ∈ [1..3.0]  (more concave, slower early)
//       This preserves the exact T60 but changes the decay *shape* correctly.
//       Attack stays crisp because S(0)=1 and we reset u on hits (no env follower lag).
//  2) Defaults at object creation: phase_random=1, bandwidth=1, micro_detune=1.
//  3) micro_detune depth increased to ±0.05 * amount for all non-fundamentals (stable, seeded).
//
// Build (macOS):
//   cc -O3 -fPIC -DPD -Wall -Wextra -Wno-unused-parameter -Wno-cast-function-type \
//     -I"/Applications/Pd-0.56-1.app/Contents/Resources/src" \
//     -arch arm64 -arch x86_64 -mmacosx-version-min=10.13 \
//     -bundle -undefined dynamic_lookup \
//     -o juicy_bank~.pd_darwin juicy_bank_tilde.c
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

#define JB_MAX_MODES 64

static t_class *juicy_bank_tilde_class;

typedef struct { unsigned int s; } jb_rng_t;
static inline void jb_rng_seed(jb_rng_t *r, unsigned int s){ if(!s) s=1; r->s = s; }
static inline unsigned int jb_rng_u32(jb_rng_t *r){ unsigned int x = r->s; x ^= x << 13; x ^= x >> 17; x ^= x << 5; r->s = x; return x; }
static inline float jb_rng_uni(jb_rng_t *r){ return (jb_rng_u32(r) >> 8) * (1.0f/16777216.0f); }
static inline float jb_rng_bi(jb_rng_t *r){ return 2.f * jb_rng_uni(r) - 1.f; }
static inline float jb_clamp(float x, float lo, float hi){ return (x<lo)?lo:((x>hi)?hi:x); }

typedef enum { DENSITY_PIVOT=0, DENSITY_INDIV=1 } jb_density_mode;

typedef struct {
    // base params
    float base_ratio, base_decay_ms, base_gain;
    float attack_ms, curve_amt, pan;
    int   active;

    // runtime
    float ratio_now, decay_ms_now, gain_now;
    float a1,a2, y1,y2, drive, env;
    float a1b,a2b, y1b,y2b;   // twin detuned partner for bandwidth
    float disp_signature;
    float micro_sig;          // stable random in [-1,1] for micro_detune (fundamental = 0)
    int   hit_gate, hit_cool;
    float y_pre_last;

    // curve shaping state
    float t60_s;              // current T60 (seconds) for this mode
    float decay_u;            // normalized time since last hit (0..1)
} jb_mode_t;

typedef struct _juicy_bank_tilde {
    t_object  x_obj; t_float f_dummy; t_float sr;
    int n_modes; jb_mode_t m[JB_MAX_MODES];
    int sel_index; float sel_index_f;

    // dispersion
    float disp_offset[JB_MAX_MODES]; float disp_target[JB_MAX_MODES];
    float dispersion, dispersion_last;

    // body globals
    float damping, brightness, position;
    float density_amt; jb_density_mode density_mode;
    float aniso, aniso_eps;
    float contact_amt, contact_sym;

    // keytrack / base
    float keytrack_amount; int keytrack_on; float basef0;

    // realism
    float phase_rand; int phase_debug;
    float bandwidth;          // 0..1, global twin-detune depth
    float micro_detune;       // 0..1, global micro-ratio detune depth (±0.05 * depth)

    // rng
    jb_rng_t rng;

    // output hygiene
    float hp_a, hpL_x1, hpL_y1, hpR_x1, hpR_y1;

    // io
    t_inlet *inR;
    t_inlet *in_damp, *in_bright, *in_pos, *in_disp, *in_density,
            *in_aniso, *in_contact, *in_idx, *in_ratio, *in_gain,
            *in_attack, *in_decay, *in_curve, *in_pan, *in_keytrack;
    t_outlet *outL, *outR;
} t_juicy_bank_tilde;

static int jb_is_near_integer(float x, float eps){ float n=roundf(x); return fabsf(x-n)<=eps; }

// Contact shaper (unchanged)
static float jb_contact_shape(float x, float amt, float sym){
    float a = 1.f + 2.f*jb_clamp(amt,0.f,1.f);
    float s = 1.f + 0.5f*jb_clamp(sym,-1.f,1.f);
    float scale = (x>=0.f)?(a*s):(a*(2.f-s));
    float z = x*scale;
    float y = tanhf(z);
    float makeup = (scale>1e-6f)?(1.f/scale):1.f;
    return y*makeup;
}

// Density mapping (unchanged)
static void jb_update_density(t_juicy_bank_tilde *x){
    float s = 1.f + 0.5f * jb_clamp(x->density_amt, -1.f, 1.f);
    if (x->density_mode == DENSITY_PIVOT){
        float r_pivot = 1.f; int fid=-1; float best=1e9f;
        for(int i=0;i<x->n_modes;i++){ if(!x->m[i].active) continue;
            float d=fabsf(x->m[i].base_ratio-1.f); if(d<best){best=d; fid=i;} }
        if (fid>=0) r_pivot = x->m[fid].base_ratio;
        for(int i=0;i<x->n_modes;i++){
            jb_mode_t *md=&x->m[i];
            if(!md->active){ md->ratio_now=md->base_ratio; continue; }
            if(i==fid) md->ratio_now = md->base_ratio;
            else md->ratio_now = r_pivot + (md->base_ratio - r_pivot) * s;
        }
    } else {
        int idxs[JB_MAX_MODES], count=0;
        for(int i=0;i<x->n_modes;i++) if(x->m[i].active) idxs[count++]=i;
        if(count==0){ for(int i=0;i<x->n_modes;i++) x->m[i].ratio_now=x->m[i].base_ratio; return; }
        for(int k=1;k<count;k++){ int id=idxs[k], j=k;
            while(j>0 && x->m[idxs[j-1]].base_ratio > x->m[id].base_ratio){ idxs[j]=idxs[j-1]; j--; }
            idxs[j]=id;
        }
        for(int j=0;j<count;j++){
            int i=idxs[j]; jb_mode_t *md=&x->m[i];
            if(j==0) md->ratio_now=md->base_ratio;
            else { int prev=idxs[j-1]; float gap=(x->m[i].base_ratio-x->m[prev].base_ratio)*s; md->ratio_now=x->m[prev].ratio_now+gap; }
        }
        for(int i=0;i<x->n_modes;i++) if(!x->m[i].active) x->m[i].ratio_now=x->m[i].base_ratio;
    }
}

static float jb_bright_gain(float ratio_rel, float b){
    float t=(jb_clamp(b,0.f,1.f)-0.5f)*2.f; float p=0.6f*t; float rr=jb_clamp(ratio_rel,1.f,1e6f);
    return powf(rr, p);
}

static float jb_position_weight(float ratio_rel, float pos){
    if (pos<=0.f) return 1.f;
    float k = roundf(jb_clamp(ratio_rel,1.f,1e6f));
    return fabsf(sinf((float)M_PI * k * jb_clamp(pos,0.f,1.f)));
}

// Compute multiplicative gain S(u) that warps linear T60 envelope into expo/log shape.
// u in [0,1], curve in [-1,1].
static inline float jb_curve_shape_gain(float u, float curve){
    if (u <= 0.f) return 1.f;
    if (u >= 1.f) return 1.f;
    float gamma;
    if (curve < 0.f){
        // exponential: faster early, slower later -> gamma < 1
        float t = -curve; // 0..1
        gamma = 1.f - t*(1.f - 0.35f);  // map to [0.35..1]
    } else if (curve > 0.f){
        // logarithmic: slower early, faster later -> gamma > 1
        float t = curve; // 0..1
        gamma = 1.f + t*(3.0f - 1.f);   // map to [1..3]
    } else {
        return 1.f;
    }
    float phi = powf(jb_clamp(u,0.f,1.f), gamma);
    float delta = phi - u;
    // 60 dB over u in [0,1] -> linear envelope = 10^{-3 u}, shaped = 10^{-3 phi}
    // multiplicative correction:
    float g = powf(10.f, -3.f * delta);
    return g;
}

static void jb_update_per_mode_gains(t_juicy_bank_tilde *x){
    for(int i=0;i<x->n_modes;i++){
        jb_mode_t *md=&x->m[i];
        if(!md->active){ md->gain_now=0.f; continue; }

        float ratio = md->ratio_now + x->disp_offset[i];
        if(i!=0) ratio += md->micro_sig * 0.05f * jb_clamp(x->micro_detune,0.f,1.f);
        if (ratio < 0.01f) ratio = 0.01f;

        float g = md->base_gain * jb_bright_gain(ratio, x->brightness);

        float a = x->aniso; float w = 1.f;
        int nearint = jb_is_near_integer(ratio, x->aniso_eps);
        if (a > 0.f){ w = (nearint ? 1.f : (1.f - a)); }
        else if (a < 0.f){ w = (!nearint ? 1.f : (1.f + a)); }
        if(w<0.f) w=0.f;

        float wp = jb_position_weight(ratio, x->position);
        md->gain_now = g * w * wp;
        if(md->gain_now<0.f) md->gain_now=0.f;
    }
}

static void jb_update_coeffs(t_juicy_bank_tilde *x){
    // smooth dispersion to target
    for(int i=0;i<x->n_modes;i++){
        float d=x->disp_target[i]-x->disp_offset[i];
        x->disp_offset[i]+=0.0025f*d;
    }

    // apply density mapping (ratio_now updated here)
    jb_update_density(x);

    // update filter coeffs per mode (including micro_detune + bandwidth twin)
    for(int i=0;i<x->n_modes;i++){
        jb_mode_t *md=&x->m[i]; if(!md->active){ md->a1=md->a2=0.f; md->a1b=md->a2b=0.f; md->t60_s=0.f; continue; }

        float ratio = md->ratio_now + x->disp_offset[i];
        if (i!=0) ratio += md->micro_sig * 0.05f * jb_clamp(x->micro_detune,0.f,1.f);
        if (i==0) ratio = md->ratio_now; // fundamental immune to dispersion & micro_detune

        float Hz = x->keytrack_on ? (x->basef0 * ratio) : ratio;
        Hz = jb_clamp(Hz, 0.f, 0.49f*x->sr);
        float w = 2.f * (float)M_PI * Hz / x->sr;

        md->decay_ms_now = md->base_decay_ms * (1.f - x->damping);
        float T60 = jb_clamp(md->decay_ms_now, 0.f, 1e7f) * 0.001f; // seconds
        md->t60_s = T60;
        float r = (T60 <= 0.f) ? 0.f : powf(10.f, -3.f / (T60 * x->sr));

        // IMPORTANT: curve shaping is now in time domain (S(u) multiplier),
        // so we keep filter radius r constant here.
        float r_eff = r;

        float c=cosf(w);
        md->a1=2.f*r_eff*c; md->a2=-r_eff*r_eff;

        // bandwidth: add a second, slightly detuned twin
        float bw = jb_clamp(x->bandwidth, 0.f, 1.f);
        if (bw > 0.f){
            float mode_scale = 0.15f + 0.85f * ((float)i / (float)(x->n_modes>1?x->n_modes-1:1));
            float detune_ppm = 1000.f * bw * mode_scale; // small
            float w2 = w * (1.f + detune_ppm * 1e-6f);
            float c2 = cosf(w2);
            md->a1b = 2.f*r_eff*c2; md->a2b = -r_eff*r_eff;
        } else {
            md->a1b=0.f; md->a2b=0.f;
        }
    }
}

static t_int *juicy_bank_tilde_perform(t_int *w){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)(w[1]);
    t_sample *inL=(t_sample *)(w[2]); t_sample *inR=(t_sample *)(w[3]);
    t_sample *outL=(t_sample *)(w[4]); t_sample *outR=(t_sample *)(w[5]);
    int n=(int)(w[6]);

    for(int i=0;i<n;i++){ outL[i]=0; outR[i]=0; }

    jb_update_coeffs(x);
    jb_update_per_mode_gains(x);

    float camt=jb_clamp(x->contact_amt,0.f,1.f);
    float csym=jb_clamp(x->contact_sym,-1.f,1.f);
    int phase_hits_block=0;

    float bw = jb_clamp(x->bandwidth, 0.f, 1.f);
    float twin_mix = 0.12f * bw; // twin contribution

    for(int m=0;m<x->n_modes;m++){
        jb_mode_t *md=&x->m[m]; if(!md->active || md->gain_now<=0.f) continue;

        float gl = sqrtf(0.5f*(1.f-md->pan)), gr = sqrtf(0.5f*(1.f+md->pan));
        float y1=md->y1, y2=md->y2, y1b=md->y1b, y2b=md->y2b, drive=md->drive, env=md->env;
        float u = md->decay_u;  // normalized decay progress

        float att_ms = jb_clamp(md->attack_ms,0.f,500.f);
        float att_a = (att_ms<=0.f)?1.f:(1.f-expf(-1.f/(0.001f*att_ms*x->sr)));
        float th = 1e-4f;

        float du = (md->t60_s > 1e-6f) ? (1.f / (md->t60_s * x->sr)) : 1.f;

        for(int i=0;i<n;i++){
            float exc=(inL[i]+inR[i]);
            float target=md->gain_now*exc;
            float abs_target=fabsf(target);

            // per-hit phase randomization + reset curve timer on hits
            if(x->phase_rand>0.f){
                if(md->hit_cool>0) md->hit_cool--;
                else {
                    if(!md->hit_gate && abs_target>1e-3f){
                        float k=x->phase_rand*0.05f*abs_target;
                        float r1=jb_rng_bi(&x->rng), r2=jb_rng_bi(&x->rng);
                        y1+=k*r1; y2+=k*r2;
                        if (bw>0.f){
                            float r3=jb_rng_bi(&x->rng), r4=jb_rng_bi(&x->rng);
                            y1b+=k*r3; y2b+=k*r4;
                        }
                        md->hit_gate=1; md->hit_cool=(int)(x->sr*0.005f);
                        u=0.f; // reset decay progress exactly at hit onset
                        phase_hits_block++;
                    }
                    if(abs_target<5e-4f) md->hit_gate=0;
                }
            }

            drive += att_a*(target-drive);

            // main
            float y_lin = (md->a1*y1 + md->a2*y2) + drive;
            y2=y1; y1=y_lin;

            float y_total = y_lin;

            // twin detuned partner (bandwidth)
            if (bw > 0.f){
                float y_lin_b = (md->a1b*y1b + md->a2b*y2b);
                y2b=y1b; y1b=y_lin_b;
                y_total += twin_mix * y_lin_b;
            }

            // Proper curve shaping: multiply by time-warp gain S(u)
            float S = jb_curve_shape_gain(u, md->curve_amt);
            y_total *= S;
            u += du; if(u>1.f) u=1.f;

            float abs_y=fabsf(y_total);
            env = env + 0.0015f*(abs_y - env);

            float y = y_total;
            if(camt>0.f && env>th){
                float mid=0.5f*(md->y_pre_last + y_total);
                float y_mid=jb_contact_shape(mid,camt,csym);
                float y_hi =jb_contact_shape(y_total,camt,csym);
                y=0.5f*(y_mid+y_hi);
            }
            md->y_pre_last = y_total;

            outL[i]+=y*gl; outR[i]+=y*gr;
        }
        md->y1=y1; md->y2=y2; md->y1b=y1b; md->y2b=y2b; md->drive=drive; md->env=env; md->decay_u=u;
    }

    // DC hygiene
    float a=x->hp_a; float x1L=x->hpL_x1, y1L=x->hpL_y1, x1R=x->hpR_x1, y1R=x->hpR_y1;
    for(int i=0;i<n;i++){
        float xl=outL[i], xr=outR[i];
        float yl=a*(y1L + xl - x1L), yr=a*(y1R + xr - x1R);
        if(fabsf(yl)<1e-20f) yl=0.f; if(fabsf(yr)<1e-20f) yr=0.f;
        outL[i]=yl; outR[i]=yr; x1L=xl; y1L=yl; x1R=xr; y1R=yr;
    }
    x->hpL_x1=x1L; x->hpL_y1=y1L; x->hpR_x1=x1R; x->hpR_y1=y1R;

    if(x->phase_debug && phase_hits_block>0){ post("juicy_bank~ phase kicks this block: %d", phase_hits_block); }
    return (w + 7);
}

// messages
static void juicy_bank_tilde_active(t_juicy_bank_tilde *x, t_floatarg idxf, t_floatarg onf){ int idx=(int)idxf-1; if(idx<0||idx>=x->n_modes) return; x->m[idx].active=(onf>0.f)?1:0; }
static void juicy_bank_tilde_modes(t_juicy_bank_tilde *x, t_floatarg nf){ int n=(int)nf; if(n<1)n=1; if(n>JB_MAX_MODES)n=JB_MAX_MODES; x->n_modes=n; }
static void juicy_bank_tilde_idx(t_juicy_bank_tilde *x, t_floatarg f){ int idx=(int)f-1; if(idx<0)idx=0; if(idx>=x->n_modes)idx=x->n_modes-1; x->sel_index=idx; x->sel_index_f=f; }
static void juicy_bank_tilde_ratio(t_juicy_bank_tilde *x, t_floatarg r){ jb_mode_t *md=&x->m[x->sel_index]; md->base_ratio=(r<=0.f)?0.01f:r; md->ratio_now=md->base_ratio; }
static void juicy_bank_tilde_gain(t_juicy_bank_tilde *x, t_floatarg g){ x->m[x->sel_index].base_gain=jb_clamp(g,0.f,1.f); }
static void juicy_bank_tilde_attack(t_juicy_bank_tilde *x, t_floatarg ms){ x->m[x->sel_index].attack_ms=(ms<0.f)?0.f:ms; }
static void juicy_bank_tilde_decay(t_juicy_bank_tilde *x, t_floatarg ms){ x->m[x->sel_index].base_decay_ms=(ms<0.f)?0.f:ms; }
static void juicy_bank_tilde_curve(t_juicy_bank_tilde *x, t_floatarg amt){ if(amt<-1.f)amt=-1.f; if(amt>1.f)amt=1.f; x->m[x->sel_index].curve_amt=amt; }
static void juicy_bank_tilde_pan(t_juicy_bank_tilde *x, t_floatarg p){ x->m[x->sel_index].pan=jb_clamp(p,-1.f,1.f); }

static void juicy_bank_tilde_freq(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    for(int i=0;i<argc && i<x->n_modes;i++){
        if(argv[i].a_type==A_FLOAT){
            float v=atom_getfloat(argv+i);
            x->m[i].base_ratio=(v<=0.f)?0.01f:v; x->m[i].ratio_now=x->m[i].base_ratio;
        }
    }
}
static void juicy_bank_tilde_decays(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    for(int i=0;i<argc && i<x->n_modes;i++){
        if(argv[i].a_type==A_FLOAT){
            float v=atom_getfloat(argv+i);
            x->m[i].base_decay_ms=(v<0.f)?0.f:v;
        }
    }
}
static void juicy_bank_tilde_amps(t_juicy_bank_tilde *x, t_symbol *s, int argc, t_atom *argv){
    (void)s;
    for(int i=0;i<argc && i<x->n_modes;i++){
        if(argv[i].a_type==A_FLOAT){
            float v=atom_getfloat(argv+i);
            x->m[i].base_gain=jb_clamp(v,0.f,1.f);
        }
    }
}

static void juicy_bank_tilde_damping(t_juicy_bank_tilde *x, t_floatarg f){ x->damping=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_brightness(t_juicy_bank_tilde *x, t_floatarg f){ x->brightness=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_position(t_juicy_bank_tilde *x, t_floatarg f){ x->position=(f<=0.f)?0.f:jb_clamp(f,0.f,1.f); }

static void juicy_bank_tilde_dispersion(t_juicy_bank_tilde *x, t_floatarg f){
    float v=jb_clamp(f,0.f,1.f);
    if(x->dispersion_last<0.f || fabsf(v-x->dispersion_last)>1e-6f){
        for(int i=0;i<x->n_modes;i++){ x->m[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng); }
        x->dispersion_last=v;
    }
    x->dispersion=v;
    for(int i=0;i<x->n_modes;i++){
        jb_mode_t *md=&x->m[i];
        if(!md->active){ x->disp_target[i]=0.f; continue; }
        if(i==0){ x->disp_target[i]=0.f; continue; }
        x->disp_target[i]=jb_clamp(md->disp_signature*x->dispersion,-1.f,1.f);
    }
}

static void juicy_bank_tilde_seed(t_juicy_bank_tilde *x, t_floatarg f){
    jb_rng_seed(&x->rng, (unsigned int)((int)f*2654435761u));
    for(int i=0;i<x->n_modes;i++){
        x->m[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng);
        x->m[i].micro_sig      =(i==0)?0.f:jb_rng_bi(&x->rng);
    }
    x->dispersion_last=x->dispersion;
    for(int i=0;i<x->n_modes;i++){
        if(!x->m[i].active || i==0){ x->disp_target[i]=0.f; continue; }
        x->disp_target[i]=jb_clamp(x->m[i].disp_signature*x->dispersion,-1.f,1.f);
    }
}

static void juicy_bank_tilde_dispersion_reroll(t_juicy_bank_tilde *x){
    for(int i=0;i<x->n_modes;i++){
        x->m[i].disp_signature=(i==0)?0.f:jb_rng_bi(&x->rng);
    }
    float d=x->dispersion; x->dispersion_last=-1.f;
    juicy_bank_tilde_dispersion(x,d);
}

// NEW: global realism messages
static void juicy_bank_tilde_bandwidth(t_juicy_bank_tilde *x, t_floatarg f){
    x->bandwidth = jb_clamp(f,0.f,1.f);
}
static void juicy_bank_tilde_micro_detune(t_juicy_bank_tilde *x, t_floatarg f){
    x->micro_detune = jb_clamp(f,0.f,1.f);
}

static void juicy_bank_tilde_density(t_juicy_bank_tilde *x, t_floatarg f){ x->density_amt=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_density_pivot(t_juicy_bank_tilde *x){ x->density_mode=DENSITY_PIVOT; }
static void juicy_bank_tilde_density_individual(t_juicy_bank_tilde *x){ x->density_mode=DENSITY_INDIV; }

static void juicy_bank_tilde_anisotropy(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso=jb_clamp(f,-1.f,1.f); }
static void juicy_bank_tilde_aniso_eps(t_juicy_bank_tilde *x, t_floatarg f){ x->aniso_eps=jb_clamp(f,0.f,0.25f); }

static void juicy_bank_tilde_contact(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_amt=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_contact_sym(t_juicy_bank_tilde *x, t_floatarg f){ x->contact_sym=jb_clamp(f,-1.f,1.f); }

static void juicy_bank_tilde_phase_random(t_juicy_bank_tilde *x, t_floatarg f){ x->phase_rand=jb_clamp(f,0.f,1.f); }
static void juicy_bank_tilde_phase_debug(t_juicy_bank_tilde *x, t_floatarg on){ x->phase_debug=(on>0.f)?1:0; }

static void juicy_bank_tilde_basef0(t_juicy_bank_tilde *x, t_floatarg f){ x->basef0=(f<=0.f)?1.f:f; }
static void juicy_bank_tilde_base_alias(t_juicy_bank_tilde *x, t_floatarg f){ juicy_bank_tilde_basef0(x,f); }
static void juicy_bank_tilde_keytrack_on(t_juicy_bank_tilde *x, t_floatarg f){ x->keytrack_on=(f>0.f)?1:0; }

static void juicy_bank_tilde_reset(t_juicy_bank_tilde *x){
    for(int i=0;i<x->n_modes;i++){
        x->m[i].active=0; x->disp_offset[i]=x->disp_target[i]=0.f;
        x->m[i].y1=x->m[i].y2=x->m[i].y1b=x->m[i].y2b=0.f;
        x->m[i].drive=x->m[i].env=0.f;
        x->m[i].hit_gate=0; x->m[i].hit_cool=0; x->m[i].y_pre_last=0.f;
        x->m[i].decay_u=0.f;
    }
    x->n_modes=1; x->sel_index=0; x->sel_index_f=1.f;
    jb_mode_t *md=&x->m[0];
    md->active=1; md->base_ratio=1.f; md->ratio_now=1.f;
    md->base_gain=0.5f; md->gain_now=0.5f;
    md->base_decay_ms=500.f; md->decay_ms_now=md->base_decay_ms*(1.f-x->damping);
    md->curve_amt=0.f; md->attack_ms=0.f; md->pan=0.f;
    md->a1=md->a2=md->a1b=md->a2b=0.f; md->t60_s=md->base_decay_ms*0.001f; md->decay_u=0.f;
}

static void juicy_bank_tilde_restart(t_juicy_bank_tilde *x){ juicy_bank_tilde_reset(x); }

static void juicy_bank_tilde_dsp(t_juicy_bank_tilde *x, t_signal **sp){
    x->sr = sp[0]->s_sr; float fc=8.f; float RC=1.f/(2.f*M_PI*fc); float dt=1.f/x->sr; x->hp_a=RC/(RC+dt);
    dsp_add(juicy_bank_tilde_perform, 6, x, sp[0]->s_vec, sp[1]->s_vec, sp[2]->s_vec, sp[3]->s_vec, sp[0]->s_n);
}

static void juicy_bank_tilde_free(t_juicy_bank_tilde *x){
    inlet_free(x->inR);
    inlet_free(x->in_damp); inlet_free(x->in_bright); inlet_free(x->in_pos);
    inlet_free(x->in_disp); inlet_free(x->in_density); inlet_free(x->in_aniso);
    inlet_free(x->in_contact); inlet_free(x->in_idx); inlet_free(x->in_ratio);
    inlet_free(x->in_gain); inlet_free(x->in_attack); inlet_free(x->in_decay);
    inlet_free(x->in_curve); inlet_free(x->in_pan); inlet_free(x->in_keytrack);
    outlet_free(x->outL); outlet_free(x->outR);
}

static void *juicy_bank_tilde_new(void){
    t_juicy_bank_tilde *x=(t_juicy_bank_tilde *)pd_new(juicy_bank_tilde_class);
    x->sr = sys_getsr(); if(x->sr<=0) x->sr=48000;

    x->n_modes=20; x->sel_index=0; x->sel_index_f=1.f;
    x->damping=0.f; x->brightness=0.5f; x->position=0.f;
    x->dispersion=0.f; x->dispersion_last=-1.f;
    x->density_amt=0.f; x->density_mode=DENSITY_PIVOT;
    x->aniso=0.f; x->aniso_eps=0.02f;
    x->contact_amt=0.f; x->contact_sym=0.f;
    x->keytrack_on=1; x->keytrack_amount=1.f; x->basef0=440.f;

    // NEW defaults as requested:
    x->phase_rand=1.f; x->phase_debug=0;
    x->bandwidth=1.f; x->micro_detune=1.f;

    jb_rng_seed(&x->rng, 0xC0FFEEu);

    x->hp_a=0.f; x->hpL_x1=x->hpL_y1=x->hpR_x1=x->hpR_y1=0.f;
    for(int i=0;i<JB_MAX_MODES;i++){ x->disp_offset[i]=x->disp_target[i]=0.f; }

    for(int i=0;i<JB_MAX_MODES;i++){
        jb_mode_t *md=&x->m[i];
        md->active=(i<20);
        md->base_ratio=(float)(i+1); md->ratio_now=md->base_ratio;
        md->base_decay_ms=500.f; md->decay_ms_now=md->base_decay_ms;
        md->base_gain=0.2f; md->gain_now=md->base_gain;
        md->attack_ms=0.f; md->curve_amt=0.f;
        if(i==0) md->pan=0.f; else md->pan=((i&1)?-0.2f:0.2f);
        md->a1=md->a2=md->y1=md->y2=md->drive=0.f; md->env=0.f;
        md->a1b=md->a2b=md->y1b=md->y2b=0.f;
        md->disp_signature = (i==0) ? 0.f : jb_rng_bi(&x->rng);
        md->micro_sig      = (i==0) ? 0.f : jb_rng_bi(&x->rng);
        md->hit_gate=0; md->hit_cool=0; md->y_pre_last=0.f;
        md->t60_s = md->base_decay_ms * 0.001f; md->decay_u=0.f;
    }

    x->inR = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_signal, &s_signal);
    x->in_damp    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("damping"));
    x->in_bright  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("brightness"));
    x->in_pos     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("position"));
    x->in_disp    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("dispersion"));
    x->in_density = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("density"));
    x->in_aniso   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("anisotropy"));
    x->in_contact = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("contact"));
    x->in_idx     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("idx"));
    x->in_ratio   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("ratio"));
    x->in_gain    = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("gain"));
    x->in_attack  = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("attack"));
    x->in_decay   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("decay"));
    x->in_curve   = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("curve"));
    x->in_pan     = inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("pan"));
    x->in_keytrack= inlet_new(&x->x_obj, &x->x_obj.ob_pd, &s_float, gensym("keytrack_on"));
    x->outL = outlet_new(&x->x_obj, &s_signal);
    x->outR = outlet_new(&x->x_obj, &s_signal);
    return (void *)x;
}

void juicy_bank_tilde_setup(void){
    juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
                           (t_newmethod)juicy_bank_tilde_new,
                           (t_method)juicy_bank_tilde_free,
                           sizeof(t_juicy_bank_tilde), CLASS_DEFAULT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);
    CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_modes, gensym("modes"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_active, gensym("active"), A_DEFFLOAT, A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_idx, gensym("idx"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_ratio, gensym("ratio"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_gain, gensym("gain"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_attack, gensym("attack"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decay, gensym("decay"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_curve, gensym("curve"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_pan, gensym("pan"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_freq, gensym("freq"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_decays, gensym("decays"), A_GIMME, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_amps, gensym("amps"), A_GIMME, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_damping, gensym("damping"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_brightness, gensym("brightness"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_position, gensym("position"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion, gensym("dispersion"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_seed, gensym("seed"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dispersion_reroll, gensym("dispersion_reroll"), 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density, gensym("density"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_pivot, gensym("density_pivot"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_density_individual, gensym("density_individual"), 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_anisotropy, gensym("anisotropy"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_aniso_eps, gensym("aniso_epsilon"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact, gensym("contact"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_contact_sym, gensym("contact_symmetry"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_random, gensym("phase_random"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_phase_debug, gensym("phase_debug"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_basef0, gensym("basef0"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_base_alias, gensym("base"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_keytrack_on, gensym("keytrack_on"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_bandwidth, gensym("bandwidth"), A_DEFFLOAT, 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_micro_detune, gensym("micro_detune"), A_DEFFLOAT, 0);

    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_reset, gensym("reset"), 0);
    class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_restart, gensym("restart"), 0);

    class_sethelpsymbol(juicy_bank_tilde_class, gensym("juicy_bank~"));
}
