# report

> A writeup describing the project, the effort that went into the
> blocks, and an analysis of any deviation from the paper (if present)

## Rayleigh Channel Simulator

To implement a Rayleigh fading channel, I've decided to implement
Smith's algorithm. The process involves generating complex Gaussian
noise for the in-phase and quadrature components of fading channel in
the frequency domain. Only half of these components get generated and
then attached to a conjugate flip to create a purely real signal in the
frequency domain. Each of these components then get scaled by a fading
spectrum representing the Doppler shift of the channel before getting
fed into a DFT. These components then get combined to form a Rayleigh
channel in the time domain. To introduce a time delay, its sufficient to
just multiply by a complex exponential to represent the phase shift at
the receiver.

Note that the implemented simulator operates under the assumption that
we're operating only at baseband.

## Creating Synthetic Data

I created simulation parameters to vary the number of sample points,
paths, channel & signal gains, Doppler shift, and SNR. For each of the
diversity techniques, I would generate unique channels, however, I would
maintain the same B-PSK encoded symbol stream. Results were collected at
various SNR values and the finally averaged over the number of
iterations to converge onto more realistic behavior of the diversity
techniques.

## Maximal-Ratio Receive Combining

There's not much to explain here! We simply take the conjugate of the
channel and multiply the received signal. This is neat as this removes
any phase-shift introduced by the channel, leaving us with only the
scaling of the channel. If the channel attenuates, the attenuation gets
halved as the amplitudes of the channel add, which plays into our favor
when decoding. Additionally, if the original channel amplified our
signal, we'd get an even higher magnitude signal, which again, helps us
with decoding in the B-PSK case.

## Alamouti Coding

Alamouti takes diversity coding to the next level by taking advantage of
different transmission channels in a constructive manner. The critical
assumption made by Alamouti is that the channel persists over two symbol
periods. In doing so, we can mitigate attenuation and still successfully
remove phase shifts. We can further improve a 2-Tx transmit system by
going up from 1-Rx to 2-Rx, essentially "adding" the benefits of MRRC to
(2-Tx, 1-Rx) Alamouti.
