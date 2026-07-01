TITLE Hole - current

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

NEURON {
	SUFFIX hole
	NONSPECIFIC_CURRENT i
	RANGE g, e
}

PARAMETER {
	g = 0.00002	(S/cm2)	<0,1e4>
	e = 0	(mV)
    : conductance is too high to neglect it in case of just loading it
}

ASSIGNED {v (mV)  i (mA/cm2)}

BREAKPOINT {
	i = g*(v - e)
}
