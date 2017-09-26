import os
from subprocess import Popen, PIPE, call 

filename = 'ahh_2.wav'
nformants = '2'
ceiling = '4500'

p = Popen(['../praat', '--run', 'formants.praat', filename, nformants, ceiling],stdout = PIPE,stderr = PIPE,stdin = PIPE)
stdout, stderr = p.communicate()
print stdout.decode().strip()
