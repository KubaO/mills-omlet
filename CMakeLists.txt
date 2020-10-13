cmake_minimum_required(VERSION 3.16)

project(mills C)
set(CMAKE_C_STANDARD 11)



add_executable(mills prog.c drand48.c)

#[[

# hyperparameters
set(param -DRS=.15 -DTR=.95 -DLR=.002 -DLD=.97 -DLE=10 -DCL=5 -DB1=.9 -DB2=.999
          -DEP=1e-8 -DWD=8e-5 -DDI=100 -DSL=200 -DN=50 -DTP=1
          -DPF='"cp%02d_%.3f"')

##############################################################
# Networks
#
# Note that temp variables have been added to force evaluation
# order.  This allows checkpoint files to be portable beetween
# different compilers.
##############################################################

# linear
set(lin       -DBK='y=x')

# perceptron
set(per       -DBK='y=T(x)')

# recurrent neural network
set(rnn       -DHS=128  -DBK='
   hp = I(HS),
   t1 = L(HS, x),
   h  = C(hp, T(A(t1, L(HS, hp)))),
   y  = h
')

# long short term memory
set(lstm   -DHS=128 -DBK='
   cp  = I(HS),
   hp  = I(HS),
   t1  = L(HS, x),
   f   = S(A(t1, L(HS, hp))),
   t2  = L(HS, x),
   i   = S(A(t2, L(HS, hp))),
   t3  = L(HS, x),
   cn  = T(A(t3, L(HS, hp))),
   t4  = M(f, cp),
   c   = C(cp, A(t4, M(i, cn))),
   t5  = L(HS, x),
   o   = S(A(t5, L(HS, hp))),
   h   = C(hp, M(o, T(c))),
   y   = h
')

# lstm with passthrough
set(lstmp -DHS=128 -DBK='
   cp  = I(HS),
   hp  = I(HS),
   t1  = L(HS, x),
   t2  = A(t1, L(HS, hp)),
   f   = S(A(CM(cp), t2)),
   t3  = L(HS, x),
   t4  = A(t3, L(HS, hp)),
   i   = S(A(CM(cp), t4)),
   t5  = L(HS, x),
   cn  = T(A(t5, L(HS, hp))),
   t6  = M(f, cp),
   c   = C(cp, A(t6, M(i, cn))),
   t7  = L(HS, x),
   t8  = A(t7, L(HS, hp)),
   o   = S(A(CM(c),  t8)),
   h   = C(hp, M(o, T(c))),
   y   = h
')

# residual lstm
set(rlstm -DHS=128 -DBK='
   cp  = I(HS),
   hp  = I(HS),
   t1  = L(HS, x),
   t2  = A(t1, L(HS, hp)),
   f   = S(A(CM(cp), t2)),
   t3  = L(HS, x),
   t4  = A(t3, L(HS, hp)),
   i   = S(A(CM(cp), t4)),
   t5  = L(HS, x),
   cn  = T(A(t5, L(HS, hp))),
   t6  = M(f, cp),
   c   = C(cp, A(t6, M(i, cn))),
   t7  = L(HS, x),
   t8  = A(t7, L(HS, hp)),
   o   = S(A(CM(c),  t8)),
   m   = L(HS, T(c)),
   h   = C(hp, M(o, A(m, L(HS, x)))),
   y   = h
')

# gated recurrent unit
set(gru    -DHS=128 -DBK='
   hp  = I(HS),
   t1  = L(HS, x),
   z   = S(A(t1, L(HS, hp))),
   t2  = L(HS, x),
   r   = S(A(t2, L(HS, hp))),
   t3  = L(HS, x),
   c   = T(A(t3, L(HS, M(r, hp)))),
   zc  = OG(1, -1, z),
   t4  = M(zc, hp),
   h   = C(hp, A(t4, M(z, c))),
   y   = h
')

# single-layer network
set(one_layer -DNW='
   x   = I(n),
   y   = L(n, MD(x))
')

# two-layer network
set(two_layer -DNW='
   x   = I(n),
   y   = L(n, MD(MD(x)))
')

# three-layer network
set(three_layer -DNW='
   x   = I(n),
   y   = L(n, MD(MD(MD(x))))
')

# Enumerated list of networks
#
lin1   = ${lin}   ${one_layer}
per1   = ${per}   ${one_layer}
rnn1   = ${rnn}   ${one_layer}
rnn2   = ${rnn}   ${two_layer}
rnn3   = ${rnn}   ${three_layer}
lstm1  = ${lstm}  ${one_layer}
lstm2  = ${lstm}  ${two_layer}
lstm3  = ${lstm}  ${three_layer}
lstmp1 = ${lstmp} ${one_layer}
lstmp2 = ${lstmp} ${two_layer}
lstmp3 = ${lstmp} ${three_layer}
rlstm1 = ${rlstm} ${one_layer}
rlstm2 = ${rlstm} ${two_layer}
rlstm3 = ${rlstm} ${three_layer}
gru1   = ${gru}   ${one_layer}
gru2   = ${gru}   ${two_layer}
gru3   = ${gru}   ${three_layer}

# default network
#
prog   = ${rnn1}

#]]
