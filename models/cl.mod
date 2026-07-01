TITLE Cl- current

UNITS {
	(mV) = (millivolt)
	(mA) = (milliamp)
	(S) = (siemens)
}

NEURON {
	SUFFIX cl
	NONSPECIFIC_CURRENT i
	RANGE g, e
}

PARAMETER {
	g = 0.	(S/cm2)	<0,1e9>
	e = -75	(mV)
    : conductance is too high to neglect it in case of just loading it
}

ASSIGNED {v (mV)  i (mA/cm2)}

BREAKPOINT {
	i = g*(v - e)
}
