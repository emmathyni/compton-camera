
import numpy as np

r_e_2 = 7.94 * 10 ** (-30)  # classical electron radius squared (according to wikipedia)
r_e = 2.8179403227 * 10 ** (-15)  # classical electron radius (according to wikipedia)

def diff_cross_section(alpha, epsilon):
    """dsigma/dOmega as defined by the klein-nishina formula
    alpha is the compton scatter angle and epsilon is the ratio of the photon energy
    divided by the electron rest energy"""
    q = 1 / (1 + epsilon*(1 - np.cos(alpha)))
    return 0.5 * r_e**2 * q**2 * (q + 1/q - np.sin(alpha)**2)


def klein_nishina_correct_alpha(epsilon, delta_angle):
    """Return a pdf and the angles for the klein-nishina distribution for alpha 0, pi
    alpha = 0 is theta = pi and therefore we need to add to get the same angle so that
    alpha=0 corresponds to the top of the unit sphere on the detector pixel
    Input: epsilon (E_photon/rest energy electron)
    delta_angle: sampling angle in the compton angle
    Output: pdf with
    pdf[i, 0] the pdf value of the klein nishina function
    pdf[i, 1] the polar angle on the unit sphere"""

    n = int(2*np.pi/delta_angle)
    alphas = np.linspace(0, np.pi, n//2)

    pdf = []
    for i in range(len(alphas)):
        p = diff_cross_section(alphas[i], epsilon) * np.sin(alphas[i]) * delta_angle**2
        pdf.append([p, alphas[i]])

    pdf = np.asarray(pdf)
    const = np.sum(pdf[:, 0])
    pdf[:, 0] = pdf[:, 0] / const
    pdf[:, 1] = np.mod(np.pi - pdf[:, 1], np.pi)

    return pdf


def main():
    kn = klein_nishina_correct_alpha(140/511, 0.5)
    print(kn)


if __name__ == '__main__':
    main()