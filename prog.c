#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "drand48.h"

// hyperparameters

#define RS .15
#define TR .95
#define LR .002
#define LD .97
#define LE 10
#define CL 5
#define B1 .9
#define B2 .999
#define EP 1e-8
#define WD 8e-5
#define DI 100
#define SL 200
#define N 50
#define TP 1
#define PF "cp%02d_%.3f"

/*
 * The network to use
 */

#define lstm2

/*
 * Predefined neural networks
 */

  // linear network
#ifdef lin1
#define Dlin
#define Done_layer
#endif

  // perceptron
#ifdef per1
#define Dper
#define Done_layer
#endif

  // recurrent neural networks
#ifdef rnn1
#define Drnn
#define Done_layer
#endif

#ifdef rnn2
#define Drnn
#define Dtwo_layer
#endif

#ifdef rnn3
#define Drnn
#define Dthree_layer
#endif

  // long short term memory
#ifdef lstm1
#define Dlstm
#define Done_layer
#endif

#ifdef lstm2
#define Dlstm
#define Dtwo_layer
#endif

#ifdef lstm3
#define Dlstm
#define Dthree_layer
#endif

  // lstm with passthrough
#ifdef lstmp1
#define Dlstmp
#define Done_layer
#endif

#ifdef lstmp2
#define Dlstmp
#define Dtwo_layer
#endif

#ifdef lstmp3
#define Dlstmp
#define Dthree_layer
#endif

  // recurrent lstm
#ifdef rlstm1
#define Drlstm
#define Done_layer
#endif

#ifdef rlstm2
#define Drlstm
#define Dtwo_layer
#endif

#ifdef rlstm3
#define Drlstm
#define Dthree_layer
#endif

  // gated recurrent unit
#ifdef gru1
#define Dgru
#define Done_layer
#endif

#ifdef gru2
#define Dgru
#define Dtwo_layer
#endif

#ifdef gru3
#define Dgru
#define Dthree_layer
#endif

/* Networks
 *
 * Note that temp variables have been added to force evaluation
 * order.  This allows checkpoint files to be portable beetween
 * different compilers.
 */

// linear network
#ifdef Dlin
#define BK y=x
#endif

// perceptron
#ifdef Dper
#define BK y=T(x)
#endif

// recurrent neural network
#ifdef Drnn
#define HS 128
#define BK                            \
   hp = I(HS),                        \
   t1 = L(HS, x),                     \
   h  = C(hp, T(A(t1, L(HS, hp)))),   \
   y  = h
#endif

// long short term memory
#ifdef Dlstm
#define HS 128
#define BK                         \
    cp = I(HS),                    \
    hp = I(HS),                    \
    t1 = L(HS, x),                 \
    f = S(A(t1, L(HS, hp))),       \
    t2 = L(HS, x),                 \
    i = S(A(t2, L(HS, hp))),       \
    t3 = L(HS, x),                 \
    cn = T(A(t3, L(HS, hp))),      \
    t4 = M(f, cp),                 \
    c = C(cp, A(t4, M(i, cn))),    \
    t5 = L(HS, x),                 \
    o = S(A(t5, L(HS, hp))),       \
    h = C(hp, M(o, T(c))),         \
    y = h
#endif

// lstm with passthrough
#ifdef Dlstmp
#define HS 128
#define BK                      \
    cp = I(HS),                 \
    hp = I(HS),                 \
    t1 = L(HS, x),              \
    t2 = A(t1, L(HS, hp)),      \
    f = S(A(CM(cp), t2)),       \
    t3 = L(HS, x),              \
    t4 = A(t3, L(HS, hp)),      \
    i = S(A(CM(cp), t4)),       \
    t5 = L(HS, x),              \
    cn = T(A(t5, L(HS, hp))),   \
    t6 = M(f, cp),              \
    c = C(cp, A(t6, M(i, cn))), \
    t7 = L(HS, x),              \
    t8 = A(t7, L(HS, hp)),      \
    o = S(A(CM(c), t8)),        \
    h = C(hp, M(o, T(c))),      \
    y = h
#endif

// residual lstm
#ifdef Drlstm
#define HS 128
#define BK                           \
    cp = I(HS),                      \
    hp = I(HS),                      \
    t1 = L(HS, x),                   \
    t2 = A(t1, L(HS, hp)),           \
    f = S(A(CM(cp), t2)),            \
    t3 = L(HS, x),                   \
    t4 = A(t3, L(HS, hp)),           \
    i = S(A(CM(cp), t4)),            \
    t5 = L(HS, x),                   \
    cn = T(A(t5, L(HS, hp))),        \
    t6 = M(f, cp),                   \
    c = C(cp, A(t6, M(i, cn))),      \
    t7 = L(HS, x),                   \
    t8 = A(t7, L(HS, hp)),           \
    o = S(A(CM(c), t8)),             \
    m = L(HS, T(c)),                 \
    h = C(hp, M(o, A(m, L(HS, x)))), \
    y = h
#endif

// gated recurrent unit
#ifdef Dgru
#define HS 128
#define BK                          \
    hp = I(HS),                     \
    t1 = L(HS, x),                  \
    z = S(A(t1, L(HS, hp))),        \
    t2 = L(HS, x),                  \
    r = S(A(t2, L(HS, hp))),        \
    t3 = L(HS, x),                  \
    c = T(A(t3, L(HS, M(r, hp)))),  \
    zc = OG(1, -1, z),              \
    t4 = M(zc, hp),                 \
    h = C(hp, A(t4, M(z, c))),      \
    y = h
#endif

// single-layer network
#ifdef Done_layer
#define NW           \
   x   = I(n),       \
   y   = L(n, MD(x))
#endif

// two-layer network
#ifdef Dtwo_layer
#define NW              \
    x = I(n),           \
    y = L(n, MD(MD(x)))
#endif

// three - layer network
#ifdef Dthree_layer
#define NW                  \
    x = I(n),               \
    y = L(n, MD(MD(MD(x))))
#endif

/*
 * The Code :)
 */

struct {
  int o, s;
} typedef V;

struct {
  int x;
  V o, i, p;
} typedef R;

int d, w, a, i, b, u, g, n, j, F, h;

R *Q, B[99], *E = B;

V I(int s) {
  V o = {d, s};
  d += s;
  return o;
}

float *p, ll = 1, *q, *e, s, *r, *G, *W, y = 1, *x, l, *z, m;

V A(V p, V i) {
  V o = I(i.s);
  *E++ = (R){4, o, p, i};
  return o;
}

V X, Y;

V C(V i, V o) {
  *E++ = (R){3, o, i};
  return o;
}

unsigned char c[256], t[sizeof c + N];

V M(V p, V i) {
  V o = I(i.s);
  *E++ = (R){5, o, p, i};
  return o;
}

struct {
  float r;
  short k, p;
  char unsigned d[sizeof c];
} k = {LR};

V S(V i) {
  V o = I(i.s);
  *E++ = (R){2, o, i};
  return o;
}

void Z(float *p, int n) {
  while (n--)
    *p++ = 0;
}

#define P(v, i)                                                                \
  n = v.o;                                                                     \
  e = G + n;                                                                   \
  p = e + v.s;                                                                 \
  q = p + d;                                                                   \
  i;                                                                           \
  while (--q, p-- > e)

V L(int s, V i) {
  V o = I(s), p = {w, s + i.s * s};
  w += p.s;
  *E++ = (R){0, o, i, p};
  return o;
}

V CM(V i) {
  V o = I(a = i.s), p = {w, a};
  w += a;
  *E++ = (R){7, o, i, p};
  return o;
}

void J(int c) {
  Z(e = G + X.o, X.s);
  c[e]++;
  for (Q = B; Q < E; ++Q) {
    P(Q->o, a = Q->i.o - n; b = Q->p.o - n; r = b + W + n) {
      { n = Q->x; }
      if (!n--) {
        *p = *r++;
        n = Q->i.s;
        while (n--)
          *p += *r++ * e[a + n];
      }
      if (!n--) {
        *p = tanh(a[p]);
      }
      if (!n--) {
        *p = 1 / (exp(-a[p]) + 1);
      }
      if (!n--) {
        p[a + u] = *p;
      }
      if (!n--) {
        *p = p[a] + b[p];
      }
      if (!n--) {
        *p = a[p] * p[b];
      }
      if (!n--) {
        *p = a[p] * Q->p.s + Q->p.o;
      }
      if (!n--) {
        *p = a[p] * *r++;
      }
    }
  }
  Z(G + d, d);
  P(Y, m = *e) m = fmax(m, *p);
  P(Y, s = 0) s += *q = exp((*p - m) / TP);
  P(Y, ) *q /= s;
}

V T(V i) {
  V o = I(i.s);
  *E++ = (R){1, o, i};
  return o;
}

int K(int K) {
  J(K);
  P(Y, s = drand48()) if ((s -= *q) < 0) break;
  return j = p - e;
}

V OG(int a, int b, V i) {
  V o = I(i.s);
  *E++ = (R){6, o, (V){b, a}};
  return o;
}

void D(char unsigned *s) {
  float *V = malloc(sizeof(float) * 2 * d);
  int R = u;
  u = V - G;
  J(c[*s++]);
  n = c[*s];
  l -= log(n[++q]--) / N;
  G += u;
  p = d + G;
  if (1 [s])
    D(s);
  else
    while (p-- > G) {
      p[x - G] = *p;
      d[p] = 0;
    }
  G -= u;
  Q = E;
  while (F < i && Q-- > B) {
    P(Q->o, a = Q->i.o - n; b = Q->p.o - n; r = b + W + n) {
      { n = Q->x; }
      if (!n--) {
        r++ [w] += *q;
        n = Q->i.s;
        z = e + a + n;
        while (n--) {
          w[r] += *q * *--z;
          d[z] += *q * *r++;
        }
      }
      if (!n--) {
        a[q] += *q * (1 - *p * *p);
      }
      if (!n--) {
        q[a] += *q * (1 - *p) * *p;
      }
      if (!n--) {
        *q += q[a + u];
      }
      if (!n--) {
        a[q] += *q;
        q[b] += *q;
      }
      if (!n--) {
        q[a] += b[p] * *q;
        b[q] += p[a] * *q;
      }
      if (!n--) {
        a[q] += Q->p.s * *q;
      }
      if (!n--) {
        a[q] += *q * *r;
        r++ [w] += *q * *p;
      }
    }
  }
  u = R;
  free(V);
}

V MD(V x) {
  V BK; //
  return y;
}

int main(int argc, char **argv) {
  FILE *f, *o = stdin;
  if (--argc) {
    if (argc > 1)
      o = fopen(2 [argv], "r");
    else
      o = NULL;
    f = fopen(1 [argv], "r");
    while (w > -(n = getc(f))) {
      ++h;
      if (!n[t]) {
        n[t]++;
        k.d[k.k] = n;
        n[c] = k.k++;
      }
    }
  }
  o &&fread(&k, sizeof k, 1, o);
  n = k.k;
  {
    V NW;
    X = x;
    Y = y;
  }
  float *R = malloc(sizeof(float) * (d + d));
  float *B = malloc(sizeof(float) * 4 * w);
  float *q = malloc(sizeof(float) * d * 2);
  G = x = R;
  W = B;
  Z(W, 4 * w);
  p = W + w;
  while (p-- > W)
    *p = 2 * drand48() * RS - RS;
  o &&fread(W, 1, sizeof B, o);
  Z(G, d);
  h /= N;
  i = h * TR;
  h -= i;
  for (t[1 + N] = m; argc; fclose(o)) {
    rewind(f);
    while (fread(t, 1 + N, 1, f)) {
      G = q;
      p = d + G;
      while (p-- > G)
        *p = p[x - G];
      D(t);
      ungetc(j = N[t], f);
      if (F++ < i) {
        a = w * 3;
        b = a - w;
        p = W + w;
        while (p-- > W) {
          m = fmax(-CL, fmin(CL, w[p]));
          ll *= B1;
          p[a] = m + (a[p] - m) * B1;
          y *= B2;
          m *= m;
          b[p] = m + (p[b] - m) * B2;
          *p += (a[p] / sqrt(p[b] / (1 - y) + EP) / (ll - 1) + WD * *p) * k.r;
          p[w] = 0;
        }
        if (1 > F % DI) {
          G = R;
          z = SL + W;
          while (z-- > B)
            putchar(k.d[K(j)]);
        }
      }
      if (1 > F % DI) {
        b = F;
        a = i;
        j = b > a;
        if (j) {
          b -= a;
          a = h;
        }
        printf(&j["\n%c%d:%d%% %f\n"], j["TV"], k.p, 100 * b / a, l / b);
      }
      l *= F != i;
    }
    char p[999];
    if (LE <= ++k.p)
      k.r *= LD;
    sprintf(p, PF, k.p, l / h);
    fwrite(&k, sizeof k, 1, o = fopen(p, "w"));
    F = l = 0;
    fwrite(B, w, sizeof *W, o);
  }
  while (putchar(k.d[K(j)])) {
  }
  free(R);
  free(B);
  free(q);
}
