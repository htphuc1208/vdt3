"""Controlled synthetic telecom RCA simulator + regime study.

Generates labelled RAN/Transport/Core incident cases with a KNOWN ground-truth
root and a realistic propagation model (topology fan-out, delay, victim
amplification, alarm floods, noise). Used to map the operating regime where each
decision mechanism (de-collapse / topology-causal / temporal-precedence /
alarm-correlation) beats the others — i.e. to predict when the method helps on
real telecom data, and to explain the OpenRCA (1-min, concentrated-signal) result.
"""
