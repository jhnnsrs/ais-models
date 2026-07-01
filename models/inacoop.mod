TITLE Cooperative Na Channels


COMMENT


published in PLoSONE, 2012
Min Huang, Maxim Volgushev, Fred Wolf 
"A small fraction of strongly cooperative sodium channels 
boosts neuronal encoding of high frequencies"

Model parameters other than cooperative channels are adapted from
Wang XJ, Buzsaki G, J Neurosci 1996, 16: 6402-6413, and
Fourcaud-Trocme N, Hansel D, van Vreeswijk C, Brunel N.
J Neurosci. 2003, 23:11628-40.


Implemented by Antonio Yanez, Uni Heidelberg
antonio.yanez@physiologie.uni-heidelberg.de
adapted from supplementary information of PLOSONE paper
Date: 20130619
Note: In general, variables are labelled according to the channels
they are operating: cooperative (_c) or non-cooperative (_nc).
The original notation labelled these variables without any sign 
(cooperative) or with a '2' at the end (non-cooperative)


ENDCOMMENT




INDEPENDENT {t FROM 0 TO 1 WITH 1 (ms)}


NEURON {
    SUFFIX nac
    USEION na READ ena WRITE ina
    
    : maximum conductance
    RANGE gbar
    
    : fraction of coop channels and their strength
    RANGE p, KJ

    : gating dynamics variables, cooperative and non-cooperative
    RANGE m_inf_c, h_inf_c
    RANGE tau_m_c, tau_h_c
    RANGE m_inf_nc, h_inf_nc
    RANGE tau_m_nc, tau_h_nc
    
    : value for time integration
    RANGE m_exp_c, h_exp_c
    RANGE m_exp_nc, h_exp_nc

    : auxiliary
    RANGE g_c, g_nc,vna
}

UNITS {
    (mA) = (milliamp)
    (mV) = (millivolt)
}

PARAMETER {
    gbar = 0.07 (mho/cm2)   : max conductance
    ena = 55 (mV)           : Na equilibrium potential
    p = 0.                 : fraction of coop. channels
    KJ = 400. (mV)          : coupling strength
    dt  (ms)                
    v (mV)
    tauf = 0.01             : factor for time constant of m, Na channels activation gating
                            : with tauf = 1 is as in original HH model 
                            : to have m instantaneously reaching m_inf as in WB model 
                            : make tauf small, say 0.01 

}

STATE {
    m_c 
    h_c
    m_nc 
    h_nc
}

ASSIGNED {

    ina (mA/cm2)
 
    : coop variables
    m_inf_c
    h_inf_c
    tau_m_c
    tau_h_c
    m_exp_c
    h_exp_c
    
    : non coop variables
    m_inf_nc
    h_inf_nc
    tau_m_nc
    tau_h_nc
    m_exp_nc
    h_exp_nc

    : auxiliary to get the final gating of each channel
    g_c
    g_nc
    vna
}

BREAKPOINT {
    SOLVE states
    g_c = p * m_c * m_c * m_c * h_c
    g_nc = ( 1. - p ) * m_nc * m_nc * m_nc * h_nc
    g_c = p * m_c * h_c
    g_nc = ( 1. - p ) * m_nc * h_nc
    ina = gbar * ( g_c + g_nc ) *  ( v - ena )

}

PROCEDURE states() {
    

    LOCAL m3
    : evaluate the equilibrium values and time constants
    evaluate_kinetics(v)

    : calulate the time constant
    m_exp_c = expdt(dt, tau_m_c)
    h_exp_c = expdt(dt, tau_h_c)
    m_exp_nc = expdt(dt, tau_m_nc)
    h_exp_nc = expdt(dt, tau_h_nc)
    
    : integration step
    m_c = m_c + m_exp_c * ( m_inf_c - m_c )
    h_c = h_c + h_exp_c * ( h_inf_c - h_c )
    m_nc = m_nc + m_exp_nc * ( m_inf_nc - m_nc )
    h_nc = h_nc + h_exp_nc * ( h_inf_nc - h_nc )
    
    : integration step literally like them
    m3 = m_inf_c * m_inf_c * m_inf_c
    m_c = m3 + ( 1 - m_exp_c ) * ( m_c - m3 )
    m3 = m_inf_nc * m_inf_nc * m_inf_nc
    m_nc = m3 + ( 1 - m_exp_nc ) * ( m_nc - m3 )
    
}

INITIAL {
    m_c = 0
    h_c = 1
    m_nc = 0
    h_nc = 1
    
    evaluate_kinetics(v)
    
    : cooperative state variables
    m_c = m_inf_c 
    h_c = h_inf_c

    : non cooperative state variables
    m_nc = m_inf_nc
    h_nc = h_inf_nc

    m_c = 0 : this is done in the matlab file to ensure spontaneouts firing at the beginning
    m_nc = 0

}

FUNCTION alpha_m (v) {
    if ( v == -35 ){
        alpha_m = 1.
    }else{
        alpha_m = -0.1 * (v + 35.)/( exp(-0.1*(v+35.)) -1 )
    }
}

FUNCTION beta_m (v) {
    beta_m = 4. * exp( -(v+60.)/18. )
}

FUNCTION alpha_h (v) {
    alpha_h = 0.35 * exp( -0.05*(v+58.) )
}

FUNCTION beta_h (v) {
    beta_h = 5. / ( 1. + exp(-0.1*(v+28.)) )
}

FUNCTION expdt(dt, tau) {
    expdt = 1. - exp(-dt/tau)
}

PROCEDURE evaluate_kinetics(v) {
    
    LOCAL a,b
    
    : cooperative
    vna = v + h_c * m_c * KJ
        
        : m kinetics
        a = alpha_m(vna)
        b = beta_m(vna)
        tau_m_c = tauf / (a + b)
        m_inf_c =  a / (a + b)

        : h kinetics
        a = alpha_h(v)
        b = beta_h(v)
        tau_h_c = 1. / (a + b)
        h_inf_c =  a / (a + b)

    : non-cooperative
        
        : m kinetics
        a = alpha_m(v)
        b = beta_m(v)
        tau_m_nc = tauf / (a + b)
        m_inf_nc =  a / (a + b)

        : h kinetics
        a = alpha_h(v)
        b = beta_h(v)
        tau_h_nc = 1. / (a + b)
        h_inf_nc =  a / (a + b)

}

