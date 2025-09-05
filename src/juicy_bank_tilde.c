// juicy_bank_tilde.c — monophonic modal **bank** (N resonators inside one object)
// Inlets (L→R):
// 1) L~ 2) R~
// 3) damping 4) brightness 5) position 6) dispersion 7) coupling 8) density 9) anisotropy 10) contact
// 11) idx 12) ratio 13) gain 14) attack 15) decay 16) pan 17) keytrack
// Outlets: 1) L~ 2) R~ 3) body_state (anything) 4) res_state (anything)
// Messages (inlet 1): base <Hz>, N <int>, aniso_P <int>, contact_soft <f>, reset, bang, debug <0|1>, bug_catch
//
// Defaults/INT: N=12, base=440, only mode 1 audible (gain=1, ratio=1, decay=600ms).
// Safety: clamp freq [1, 0.45*fs], r<=0.999999, pan[-1,1]. Attack=0ms => instant (no mute).
// Mix normalization: mix_scale = 1/sqrt(n_active).
#include "m_pd.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif
#define MAXN 128
static t_class *juicy_bank_tilde_class;
typedef struct {
float ratio, gain, attack_ms, decay_ms, pan, keytrack;
double a1, a2, y1, y2, env, gl, gr, freq_hz, rcur;
float amp_w, r_w, pos_w, aniso_w;
unsigned char active, dirty_coeffs;
} t_mode;
typedef struct _juicy_bank_tilde {
t_object x_obj;
t_float f_dummy;
// body
int N;
float base_hz;
float damping, brightness, position, dispersion, coupling, density, anisotropy, contact;
int aniso_P;
float contact_soft, couple_df;
// derived
float r_mul, bright_slope, aniso_gamma, mix_scale;
int n_active;
// edit cursor
int edit_idx;
// modes
t_mode m[MAXN];
// dsp
float sr;
int debug;
// outlets
t_outlet *outL, *outR, *outBody, *outRes;
} t_juicy_bank_tilde;
static inline float clampf(float x, float lo, float hi){ if(x<lo) return lo; if(x>hi) return hi; return x; }
static inline double denorm_fix(double v){ return (v<1e-30 && v>-1e-30)?0.0:v; }
static void bank_recalc_body(t_juicy_bank_tilde *x){
x->damping = clampf(x->damping, 0.f, 1.f);
x->brightness = clampf(x->brightness, -1.f, 1.f);
x->position = clampf(x->position, 0.f, 1.f);
x->dispersion = clampf(x->dispersion, -0.5f, 0.5f);
x->coupling = clampf(x->coupling, 0.f, 1.f);
x->density = clampf(x->density, 0.f, 0.5f);
x->anisotropy = clampf(x->anisotropy, -1.f, 1.f);
x->contact = clampf(x->contact, 0.f, 1.f);
if (x->base_hz <= 0) x->base_hz = 440.f;
if (x->N < 1) x->N = 1;
if (x->N > MAXN) x->N = MAXN;
if (x->aniso_P < 1) x->aniso_P = 2;
if (x->contact_soft < 0.5f) x->contact_soft = 2.f;
if (x->couple_df <= 0) x->couple_df = 200.f;
x->r_mul = 1.f - 0.5f * x->damping; // linear (Elements-inspired)
x->aniso_gamma = x->anisotropy;
x->bright_slope = x->brightness;
}
static void mode_update_pan(t_mode *m){
double p = clampf(m->pan, -1.f, 1.f);
double th = (p + 1.0) * 0.25 * M_PI; // [-1,1] -> [0, pi/2]
m->gl = cos(th);
m->gr = sin(th);
}
static void bank_update_coeffs_one(t_juicy_bank_tilde *x, int k){
if (k < 1 || k > x->N) return;
t_mode *m = &x->m[k-1];
int N = x->N;
double base = (x->base_hz>0?x->base_hz:1.0);
double fr = (m->keytrack!=0.f) ? base * (double)m->ratio : (double)m->ratio;
if (fr < 1.0) fr = 1.0;
fr *= (1.0 + (double)x->dispersion * (double)(k*k));
double s = sin(M_PI * (double)k / (double)N);
fr *= (1.0 + (double)x->density * s);
double fs = (double)x->sr;
if (fr > 0.45*fs) fr = 0.45*fs;
m->freq_hz = fr;
double d_s = (m->decay_ms<=0?0.0:m->decay_ms*0.001);
double r = (d_s<=0.0)? 0.0 : exp(-1.0/(d_s*fs));
double frel = fr / base; if (frel < 0.1) frel = 0.1;
double expo = 1.0 + 0.8 * sqrt(frel);
double r_w = pow(clampf(x->r_mul,0.f,1.f), expo);
r *= r_w;
if (r>0.999999) r = 0.999999;
if (r<0.0) r = 0.0;
m->rcur = r;
double w = 2.0 * M_PI * fr / fs;
m->a1 = 2.0 * r * cos(w);
m->a2 = - (r * r);
double posw = fabs(sin(M_PI * (double)k * (double)clampf(x->position,0.f,1.f)));
if (posw < 0.05) posw = 0.05;
int P = (x->aniso_P<1?1:x->aniso_P);
double aniso = 1.0 + (double)x->aniso_gamma * cos(2.0*M_PI * (double)(k-1) / (double)P);
if (aniso < 0.25) aniso = 0.25; if (aniso > 2.0) aniso = 2.0;
if (aniso < 0.25)
    aniso = 0.25;
else if (aniso > 2.0)
    aniso = 2.0;
double bright = pow(fr/base, (double)clampf(x->bright_slope, -2.f, 2.f));
if (bright < 0.05) bright = 0.05; if (bright > 20.0) bright = 20.0;
if (bright < 0.05)
    bright = 0.05;
else if (bright > 20.0)
    bright = 20.0;

m->pos_w = (float)posw;
m->aniso_w = (float)aniso;
@@ -214,21 +220,24 @@ double env=m->env;
double fs = (double)x->sr;
double att = (m->attack_ms<=0? 0.0 : exp(-1.0/( (double)m->attack_ms*0.001 * fs )));
double dec = (m->decay_ms <=0? 0.0 : exp(-1.0/( (double)m->decay_ms *0.001 * fs )));
double gl=m->gl, gr=m->gr;
double amp = (double)m->gain * (double)x->mix_scale * (double)m->amp_w * (double)m->aniso_w;

for (int i=0;i<n;++i){
double xin = 0.5*((double)inL[i] + (double)inR[i]) * (double)m->pos_w;
double tgt = (fabs(xin) > 1e-9) ? 1.0 : 0.0;
if (tgt > env) env = tgt + (env - tgt) * att; else env = tgt + (env - tgt) * dec;
double y0 = a1*y1 + a2*y2 + xin;
y2 = y1; y1 = y0;
double ynl = contact_shaper(y0, c_amt, c_soft);
double yamp= amp * env * ynl;
outL[i] = (t_sample)denorm_fix( (double)outL[i] + yamp * gl );
outR[i] = (t_sample)denorm_fix( (double)outR[i] + yamp * gr );
}
m->y1=y1; m->y2=y2; m->env=env;
    double gl=m->gl, gr=m->gr;
    double amp = (double)m->gain * (double)x->mix_scale;

    for (int i=0;i<n;++i){
        double xin = 0.5*((double)inL[i] + (double)inR[i]) * (double)m->amp_w;
        xin = contact_shaper(xin, c_amt, c_soft);
        double tgt = (fabs(xin) > 1e-9) ? 1.0 : 0.0;
        if (tgt > env)
            env = tgt + (env - tgt) * att;
        else
            env = tgt + (env - tgt) * dec;
        double y0 = a1*y1 + a2*y2 + xin;
        y2 = y1; y1 = y0;
        double yamp = amp * env * y0;
        outL[i] = (t_sample)denorm_fix( (double)outL[i] + yamp * gl );
        outR[i] = (t_sample)denorm_fix( (double)outR[i] + yamp * gr );
    }
    m->y1=y1; m->y2=y2; m->env=env;
}

return (t_int *)(w+7);
@@ -352,7 +361,7 @@ static void *juicy_bank_tilde_new(t_symbol *s, int argc, t_atom *argv){
(void)s;
t_juicy_bank_tilde *x = (t_juicy_bank_tilde*)pd_new(juicy_bank_tilde_class);
x->N=12; x->base_hz=440;
x->damping=0; x->brightness=0; x->position=0.5f; x->dispersion=0;
x->damping=0; x->brightness=0; x->position=0.37f; x->dispersion=0;
x->coupling=0; x->density=0; x->anisotropy=0; x->contact=0;
x->aniso_P=2; x->contact_soft=2; x->couple_df=200;
x->edit_idx=1; x->sr=44100; x->debug=0; x->n_active=0; x->mix_scale=1.f;
for (int i=0;i<argc;i++){
if (argv[i].a_type==A_SYMBOL){
const char *k = atom_getsymbol(argv+i)->s_name;
if ((!strcmp(k,"@N") || !strcmp(k,"N")) && i+1<argc && argv[i+1].a_type==A_FLOAT){
int n=(int)atom_getfloat(argv+i+1); if (n<1) n=1; if (n>MAXN) n=MAXN; x->N=n; i++;
} else if ((!strcmp(k,"@base") || !strcmp(k,"base")) && i+1<argc && argv[i+1].a_type==A_FLOAT){
float b=atom_getfloat(argv+i+1); if (b>0) x->base_hz=b; i++;
} else if ((!strcmp(k,"@aniso_P") || !strcmp(k,"aniso_P")) && i+1<argc && argv[i+1].a_type==A_FLOAT){
int p=(int)atom_getfloat(argv+i+1); if (p<1) p=1; x->aniso_P=p; i++;
} else if ((!strcmp(k,"@contact_soft") || !strcmp(k,"contact_soft")) && i+1<argc && argv[i+1].a_type==A_FLOAT){
float cs=atom_getfloat(argv+i+1); if (cs<0.5f) cs=0.5f; if (cs>10.f) cs=10.f; x->contact_soft=cs; i++;
}
}
}
bank_recalc_body(x);
for (int k=1; k<=x->N; ++k){
t_mode *m=&x->m[k-1];
m->ratio=1.f; m->gain=0.f; m->attack_ms=0.f; m->decay_ms=600.f; m->pan=0.f; m->keytrack=1.f;
m->a1=m->a2=m->y1=m->y2=m->env=0.0; m->gl=1.0; m->gr=0.0; m->freq_hz=0.0; m->rcur=0.0;
m->amp_w=1.f; m->r_w=1.f; m->pos_w=1.f; m->aniso_w=1.f;
m->active=0; m->dirty_coeffs=1;
bank_update_coeffs_one(x,k);
}
// INT audible mode
x->m[0].gain=1.f; x->m[0].ratio=1.f; x->m[0].decay_ms=600.f; x->edit_idx=1;
// outlets
x->outL = outlet_new(&x->x_obj, &s_signal);
x->outR = outlet_new(&x->x_obj, &s_signal);
x->outBody = outlet_new(&x->x_obj, &s_anything);
x->outRes = outlet_new(&x->x_obj, &s_anything);
// extra inlets
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("signal"), gensym("signal"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("damping"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("brightness"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("position"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("dispersion"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("coupling"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("density"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("anisotropy"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("contact"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("idx"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("ratio"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("gain"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("attack"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("decay"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("pan"));
inlet_new(&x->x_obj, &x->x_obj.ob_pd, gensym("float"), gensym("keytrack"));
bank_emit_body_state(x);
bank_emit_res_selected(x);
return (void*)x;
}
void juicy_bank_tilde_setup(void){
juicy_bank_tilde_class = class_new(gensym("juicy_bank~"),
(t_newmethod)juicy_bank_tilde_new, 0,
sizeof(t_juicy_bank_tilde), CLASS_DEFAULT, A_GIMME, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)juicy_bank_tilde_dsp, gensym("dsp"), A_CANT, 0);
CLASS_MAINSIGNALIN(juicy_bank_tilde_class, t_juicy_bank_tilde, f_dummy);
class_addmethod(juicy_bank_tilde_class, (t_method)set_damping, gensym("damping"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_brightness, gensym("brightness"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_position, gensym("position"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_dispersion, gensym("dispersion"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_coupling, gensym("coupling"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_density, gensym("density"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_anisotropy, gensym("anisotropy"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_contact, gensym("contact"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_idx, gensym("idx"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_ratio, gensym("ratio"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_gain, gensym("gain"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_attack, gensym("attack"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_decay, gensym("decay"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_pan, gensym("pan"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)set_keytr, gensym("keytrack"),A_FLOAT,0);
class_addmethod(juicy_bank_tilde_class, (t_method)msg_base, gensym("base"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)msg_N, gensym("N"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)msg_anisoP, gensym("aniso_P"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)msg_contact_soft, gensym("contact_soft"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)msg_reset, gensym("reset"), 0);
class_addmethod(juicy_bank_tilde_class, (t_method)msg_bang, gensym("bang"), 0);
class_addmethod(juicy_bank_tilde_class, (t_method)msg_debug, gensym("debug"), A_FLOAT, 0);
class_addmethod(juicy_bank_tilde_class, (t_method)msg_bug_catch, gensym("bug_catch"), 0);
