import matplotlib.colors
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
from kleinnishina import *
from scipy.special import sph_harm
from wigner import wigner_dl
from scipy.interpolate import griddata
from scipy.stats import vonmises_fisher

import array_api_compat.numpy as xp
# import array_api_compat.cupy as xp
# import array_api_compat.torch as xp
import parallelproj
from array_api_compat import to_device, device

# choose a device (CPU or CUDA GPU) (for the parallelproj library)
if 'numpy' in xp.__name__:
    # using numpy, device must be cpu
    dev = 'cpu'
elif 'cupy' in xp.__name__:
    # using cupy, only cuda devices are possible
    dev = xp.cuda.Device(0)
elif 'torch' in xp.__name__:
    # using torch valid choices are 'cpu' or 'cuda'
    dev = 'cuda'

# global parameters
Rd = 40  # cm, radius of detector
Rs = 1  # cm, radius of source/lesion
R = 8  # cm, radius of head
delta = 0.1  # voxel length in same unit as the Rd, Rs, R
pixel_area = 1  # cm^2, area of detector pixel
T = 1  # s detection time
eps = 1  # detection efficiency
A_dose = 0.37 * 10 ** 9  # typical activity of a dose administered (in Bq)
As = 0.9 * A_dose  # source activity in Bq
Ab = 0.1 * A_dose  # background activity in Bq
E = 140  # energy of the photons in keV
epsilon = E / 511  # energy of the photon compared to the rest energy of electron


def activity(x, y, z, voxel_size, source):
    """Returns the activity/volume in position x,y,z with and without a tumor/source"""
    # activity per volume should be the same in both cases
    # generates voxels in a cube that the head and source/lesion is in
    r2 = x**2 + y**2 + z**2
    if source:
        if r2 <= Rs**2:
            return As/voxel_size
        elif Rs**2 < r2 <= R**2:
            return Ab/voxel_size
        else:
            return 0
    else:
        #A = (R-Rs)*Ab/R + Rs*As/R
        A = ((R/Rs)**3 * As + Ab * R**3/(R**3-Rs**3))
        if r2 <= R**2:
            return A/voxel_size
        else:
            return 0


def activity_diff_size(x, y, z, voxel_size, source):
    """function for determining voxel activity for two different size lesions, not used can be developed"""
    #source=True gives the smaller tumor, source=False gives the bigger tumor
    r2 = x ** 2 + y ** 2 + z ** 2
    R_bigger = 3*Rs
    if source:
        if r2 <= Rs ** 2:
            return As / voxel_size
        elif Rs ** 2 < r2 <= R ** 2:
            return Ab / voxel_size
        else:
            return 0
    else:
        As_bigger = As * (Rs/R_bigger)**3
        Ab_bigger = Ab
        if r2 <= R_bigger ** 2:
            return As_bigger / voxel_size
        elif R_bigger**2 < r2 <= R**2:
            return Ab_bigger/voxel_size
        else:
            return 0


def create_voxels(source):
    """Creates the voxels with the corresponding activity per volume.
    input: source (True or False)
    output: voxels f
            image dimensions img_dim, shape (n, n, n)
            image origin img_origin shape (3,)
            voxel size voxel_size shape (3, )
            """
    # setup the image dimensions
    n = int(2*R/delta)
    img_dim = (n, n, n)

    # define the voxel sizes (in physical units)
    voxel_size = to_device(xp.asarray([delta, delta, delta], dtype=xp.float32), dev)
    # define the origin of the image (location of voxel (0,0,0) in physical units)
    img_origin = to_device(xp.asarray([-R, -R, -R], dtype=xp.float32), dev) + 0.5 * voxel_size

    x = np.linspace(-R, R, n, endpoint=False) + delta/2  # voxels defined by their center point
    y = np.linspace(-R, R, n, endpoint=False) + delta/2
    z = np.linspace(-R, R, n, endpoint=False) + delta/2
    f = np.zeros((n, n, n), dtype='float32')
    for i in range(n):
        for j in range(n):
            for k in range(n):
                f[i, j, k] = activity(x[i], y[j], z[k], delta ** 3, source)
                #f[i, j, k] = activity_diff_size(x[i], y[j], z[k], delta**3, source)

    return f, img_dim, img_origin, voxel_size


def scale_voxels(gamma, zd, img, img_origin, voxel_size):
    """Scale each voxel value according to the inverse squared distance between the voxel and the detector pixel
    Input:
    gamma, angle of position for detector, radians
    zd: position of height on cylinder detector, cm
    img: the voxels with the activity/volume
    img_origin: vector, position of origin
    voxel_size: side of voxels, cm"""

    pos_detector = np.asarray([Rd * np.cos(gamma), Rd * np.sin(gamma), zd])

    scaled_img = img.copy()
    for i in range(scaled_img.shape[0]):
        for j in range(scaled_img.shape[1]):
            for k in range(scaled_img.shape[2]):
                voxel_pos = img_origin + np.asarray([i, j, k]) * voxel_size
                dist_2 = pos_detector - voxel_pos
                dist_2 = np.multiply(dist_2, dist_2)
                dist_2 = np.sum(dist_2)

                scaled_img[i, j, k] = scaled_img[i, j, k] / dist_2

    return scaled_img


def x(z_prime, theta, phi, gamma, z):
    """Get x-coordinate on line parametrized by z_prime from detector pixel at position defined by gamma, z
    in the direction of theta and phi"""
    return - z_prime * np.tan(theta) * np.sin(gamma) * np.sin(phi) + np.cos(gamma) * (Rd - z_prime)


def y(z_prime, theta, phi, gamma, z):
    """Get y-coordinate on line parametrized by z_prime from detector pixel at position defined by gamma, z
        in the direction of theta and phi"""
    return z_prime * np.tan(theta) * np.cos(gamma) * np.sin(phi) + np.sin(gamma) * (Rd - z_prime)


def z(z_prime, theta, phi, gamma, zd):
    """Get z-coordinate on line parametrized by z_prime from detector pixel at position defined by gamma, z
        in the direction of theta and phi"""
    return zd + z_prime * np.tan(theta) * np.cos(phi)


def find_LOR_coordinates(theta, phi, gamma, zd):
    """Return the LOR end and start in world coordinates
    Line is defined by the start position at the detector pixel position and has the direction of theta, phi"""
    start_cor = [x(Rd-R, theta, phi, gamma, zd), y(Rd-R, theta, phi, gamma, zd), z(Rd-R, theta, phi, gamma, zd)]
    end_cor = [x(Rd + R, theta, phi, gamma, zd), y(Rd + R, theta, phi, gamma, zd), z(Rd + R, theta, phi, gamma, zd)]

    return np.asarray(start_cor), np.asarray(end_cor)


def fwd_proj(gamma, zd, delta_ang, img, img_dim, img_origin, voxel_size):
    """Function for performing the forward projection from the voxels to the detector
    Input:
    delta_ang: in radians, the sampling angle that we use in the polar direction
    the rest are explained in other functions
    Output:
    n_photons: the forward projection results as an array with
    n_photons[i, 0]: the forward projection on the line of response specified by theta and phi below
    n_photons[i, 1]: the polar angle theta for the line of response
    n_photons[i, 2]: the azimuthal angle phi for the line of response
    """
    n = int(np.pi/(2*delta_ang))
    thetas = np.linspace(0, np.pi/2, n)

    n_photons = []

    for i in range(len(thetas)):
        # generate more even sampling on the sphere by adjusting the number of azimuthal angles we sample
        if thetas[i] != 0:
            m = int(np.sin(thetas[i])*2*np.pi/delta_ang)
            phis = np.linspace(0, 2 * np.pi, m)
        else:
            phis = [0]

        for j in range(len(phis)):
            t = thetas[i]
            p = phis[j]
            xstart, xend = find_LOR_coordinates(t, p, gamma, zd)

            # do forward projection
            # need to multiply fwd_proj with T*pixel_area*eps*cos(theta)
            forward_proj = parallelproj.joseph3d_fwd(xstart, xend, img, img_origin, voxel_size)
            photons = forward_proj * T * pixel_area * eps * np.cos(t)
            res = [photons, t, p]
            n_photons.append(res)

    n_photons = np.asarray(n_photons)

    return n_photons  # shape (n*n//4, 3)


def kn_distribution(pdf_val, theta, delta_angle):
    """Return the distribution of photons in rings around the sphere by extending the klein nishina
    distribution to the azimuthal angles (beta), it is uniformly distributed over beta
    Input:
        pdf_val: the value of the pdf at the compton scatter angle
        theta: the polar angle/compton scatter angle
        delta_angle: the sampling angle
    Output:
        distribution: the pdf_val/the number of points for the rings for each azimuthal angle beta"""
    # if the angle is the same for every alpha then the size between betas is different for each latitude
    if theta == 0:
        length = 1
    else:
        length = int(2*np.pi*np.sin(theta)/delta_angle)

    betas = np.linspace(0, 2*np.pi, length)
    distribution = np.ones((length, ))
    for i in range(len(distribution)):
        distribution[i] = pdf_val / length

    assert abs(np.sum(distribution) - pdf_val) < 10**(-3)
    distribution = np.column_stack((distribution, betas))

    return distribution


def cartesian_to_spherical(pos):
    """Convert Cartesian coordinates (x, y, z) to spherical coordinates (r, θ, φ)."""
    x, y, z = pos
    r = np.sqrt(x**2 + y**2 + z**2)
    assert abs(z) < 1 + 10**(-10)
    assert abs(r-1) < 10 ** (-10)
    theta = np.arccos(z / r)
    phi = np.arctan2(y, x)
    return theta, phi


def spherical_to_cartesian(theta, phi):
    """Take spherical coordinates and return their cartesian coordinates"""
    return np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)


def get_alpha_distribution(photons, theta, phi, kappa, delta_ang):
    """Function for returning the distribution when using the von mises fisher (modified version)
    to blur the measurements
    Input:
        photons: the number of photons for a measurement theta, phi
        theta: polar angle on detector
        phi: azimuthal angle on detector
        kappa: concentration parameter for the von mises fisher distribution
        delta_ang: sampling angle
    Output:
        vecs: the cartesian positions
        photons*pdf: the blurred measurement of photons for the positions vecs"""
    # data is an array with #photons for a specific theta and phi and this should be multiplied with a von Mises Fisher
    # distribution
    # should return the cartesian coordinates and the associated value of the number of photons

    mu = np.array([np.cos(phi)*np.sin(theta), np.sin(theta)*np.sin(phi), np.cos(theta)])  # mean direction of the vmf
    vmf = vonmises_fisher(mu, kappa)  # is von Mises Fisher the correct distribution?

    # generate vectors for the circle which has a specific azimuthal angle phi but varies in polar angle
    n = int(np.pi/delta_ang)
    ts = np.linspace(0, np.pi, n)
    vecs = []
    for t in ts:
        vecs.append([np.sin(t)*np.cos(phi), np.sin(t)*np.sin(phi), np.cos(t)])
        vecs.append([np.sin(t)*np.cos(phi + np.pi), np.sin(t)*np.sin(phi + np.pi), np.cos(t)])

    # generate the pdf
    pdf = np.asarray(vmf.pdf(vecs))
    pdf = pdf/np.sum(pdf)
    vecs = np.asarray(vecs)
    plot = False
    if plot:
        #plot
        fig, ax = plt.subplots(subplot_kw={'projection': "3d"})
        pdfnorm = matplotlib.colors.Normalize(vmin=pdf.min(), vmax=pdf.max())
        ax.scatter(vecs[:, 0], vecs[:, 1], vecs[:, 2], facecolors=plt.cm.viridis(pdfnorm(pdf)))
        #plt.colorbar()
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')
        ax.set_aspect('equal')
        plt.show()

    assert abs(np.sum(pdf) - 1) < 10**(-10)

    return vecs, photons * pdf


def rotation_on_unit_sphere(x, theta, phi):
    """Return the rotation of the vector(s) x on the unit sphere with first a rotation of theta around x axis (angle
    between z-axis is now theta) and then a rotation of phi around the z axis (azimuthal angle is now phi)."""
    # x must be an array and have shape (3, n)

    rx = np.array([[1, 0, 0],
                   [0, np.cos(theta), -np.sin(theta)],
                   [0, np.sin(theta), np.cos(theta)]])

    rz = np.array([[np.cos(phi), -np.sin(phi), 0],
                  [np.sin(phi), np.cos(phi), 0],
                  [0, 0, 1]])

    rot = np.matmul(rz, rx)

    return np.matmul(rot, x)


def detector_not_perfect(n_photons, kn_dist, delta_ang, kappa):
    """n_photons is (n, 3) from fwd proj, kn_dist is now kn_dist over alpha
    Perform calculations on the input (forward projection measurements of shape (n, 3) to the output
    values Lambda(theta, phi, compton scatter angle=alpha)
    Input:
        n_photons: forward projection measurements, shape (n, 3)
        kn_dist: the klein nishina distribution for a ring around the sphere
        delta_ang: sampling angle in compton and polar angle
        kappa: list with concentration parameters for the vmf
    Output:
        new_photon_data which is Lambda(theta, phi, compton scatter angle=alpha) of the shape (n, 3)
        with [i, 0] the value of the #photons for the angle polar angle [i, 1] and azimuthal angle [i, 2]"""
    global_coor_list = []
    global_values_list = []

    for i in range(n_photons.shape[0]):
        print((i+1)/len(n_photons), ' fraction of loops started')
        timing = time.time()
        photons, t, p = n_photons[i, :]
        photon_dist_alpha = photons * kn_dist[:, 0]  # get the compton scattered values over alpha

        photon_dist_list = []  # want this to have shape (m, 3) with 0 #photons, 1 alpha, 2 beta
        for j in range(photon_dist_alpha.shape[0]):
            # for each alpha value, get the distribution over beta
            beta_dist = kn_distribution(photon_dist_alpha[j], kn_dist[j, 1], delta_ang)
            for k in range(len(beta_dist)):
                photon_dist_list.append([beta_dist[k, 0], kn_dist[j, 1], beta_dist[k, 1]])
                #print(photon_dist_list[-1], 'photon list dist')

        photon_dist_list = np.asarray(photon_dist_list)

        coordinates = []
        values = []
        for j in range(photon_dist_list.shape[0]):
            coors, vals = get_alpha_distribution(photon_dist_list[j, 0], photon_dist_list[j, 1], photon_dist_list[j, 2],
                                                 kappa, delta_ang)
            coordinates.append(coors)
            values.append(vals)

        # do rotation
        # perform some reshaping of the arrays to match the format expected by matrix rotation
        coordinates = np.asarray(coordinates)
        values = np.asarray(values)
        coordinates = np.reshape(coordinates, (-1, 3))
        values = np.reshape(values, (-1))

        # rotate all values to the correct incident angle t, p
        # was easier to think of in cartesian coordinates
        new_coordinates = rotation_on_unit_sphere(np.transpose(coordinates), t, p)
        norms = np.linalg.norm(new_coordinates, axis=0)
        for n in norms:
            assert abs(n-1) < 10**(-10)
        # add new coordinates and values to global list
        global_values_list.append(values)
        global_coor_list.append(np.transpose(new_coordinates))
        print(time.time()-timing, 'time to complete one measurement')

    global_values_list = np.asarray(global_values_list)
    global_coor_list = np.asarray(global_coor_list)
    # reshape global lists
    global_values_list = np.reshape(global_values_list, (-1))
    global_coor_list = np.reshape(global_coor_list, (-1, 3))

    # return to spherical coordinates
    new_angles = np.asarray([cartesian_to_spherical(global_coor_list[i, :]) for i in range(global_coor_list.shape[0])])

    new_photon_data = np.column_stack((global_values_list, new_angles))

    return new_photon_data


def detectability_first_step(gamma, zd, delta_ang):
    """Return the forward projection of the delta image and the image (background)
    input: detector position defined by gamma, zd and the sampling angle delta_ang
    for the measurements on the detector"""
    # create image only once
    t = time.time()

    img_s, img_dim, img_origin, voxel_size = create_voxels(True)
    img_ns, _, _, _ = create_voxels(False)
    img = img_s - img_ns

    #print(time.time() - t, 'create voxels')

    t = time.time()
    # scale voxels only once
    img = scale_voxels(gamma, zd, img, img_origin, voxel_size)
    img_ns = scale_voxels(gamma, zd, img_ns, img_origin, voxel_size)

    #print(time.time() - t, 'scale voxels')
    # do forward projection only once
    t = time.time()
    n_photons_delta = fwd_proj(gamma, zd, delta_ang, img, img_dim, img_origin, voxel_size)
    #print(n_photons_delta)

    n_photons_bg = fwd_proj(gamma, zd, delta_ang, img_ns, img_dim, img_origin, voxel_size)
    #print(time.time() - t, 'fwd proj')

    # remove measurements where the number of photons is zero
    #mask = n_photons_delta[:, 0] != 0
    #n_photons_delta = n_photons_delta[mask]

    #mask2 = n_photons_bg[:, 0] != 0
    #n_photons_bg = n_photons_bg[mask2]

    np.save(f'fwd_da{delta_ang}_bg.npy', n_photons_bg)
    np.save(f'fwd_da{delta_ang}_diff.npy', n_photons_delta)
    return n_photons_delta, n_photons_bg


def wrapped_gaussian(x, mu, sigma):
    """Return the value of wrapped Gaussian at x with mean mu and standard deviation sigma"""
    # this series yields a good approximation for small sigma
    # https://tisl.cs.toronto.edu/publication/201410-sdf-wrapped_normal_evaluation/201410-sdf-wrapped_normal_evaluation.pdf
    kmax = 50  # paper seems to suggest for small sigma you only need about kmax=10 to get an accuracy of 10**(-15)
                # when sigma = 0.5
    ks = np.arange(-kmax, kmax, 1)  # generate integers from k=-inf to k=inf
    prob = 0
    for k in ks:
        prob += np.exp(-(x - mu + 2 * np.pi * k) ** 2 / (2 * sigma ** 2))
    prob = prob / (sigma * np.sqrt(2 * np.pi))
    return prob


def detectability_kappa(gamma, zd, delta_ang, kappas):
    """ Simulation of the detected values for the non perfect detector
    Save the calculated values Lambda(theta, phi, alpha) for each kappa in the vmf distribution
    delta_ang: the sampling angle in the polar direction
    alpha is the compton scatter angle"""
    # load the forward proj for delta f and f bg
    #n_photons_delta, n_photons_bg = detectability_first_step(gamma, zd, delta_ang)
    n_photons_delta = np.load(f'fwd_da{delta_ang}_diff.npy')
    n_photons_bg = np.load(f'fwd_da{delta_ang}_big.npy')
    # remove zeros using mask
    mask = n_photons_delta[:, 0] != 0
    n_photons_delta = n_photons_delta[mask]
    mask = n_photons_bg[:, 0] != 0
    n_photons_bg = n_photons_bg[mask]

    kn_dist = klein_nishina_correct_alpha(epsilon, delta_ang)  # get the compton distribution in the angle alpha

    assert (abs(np.sum(kn_dist[:, 0]) - 1) < 10**(-8))  # assert that we have a probability distribution
    print(np.sum(n_photons_delta[:, 0]), 'fwd proj delta sum of n_photons')
    print(np.sum(n_photons_bg[:, 0]), 'fwd proj bg sum of n_photons')

    for k in kappas:
        # get the delta Lambda(theta, phi, alpha)
        values_d = detector_not_perfect(n_photons_delta, kn_dist, delta_ang, k)
        print(np.sum(values_d[:, 0]), 'sum of all bins, values delta')
        np.save(f'vd_kappa{k}_da{delta_ang}.npy', values_d)

        # get the Lambda(theta, phi, alpha) for the background case
        values_bg = detector_not_perfect(n_photons_bg, kn_dist, delta_ang, k)
        print(np.sum(values_bg[:, 0]), 'sum of all bins values bg')
        np.save(f'vbg_kappa{k}_da{delta_ang}.npy', values_bg)


def perfect_detector(fwd_proj_measurements, delta_ang):
    """Function for calculating the Lambda values for the perfect compton scatter detector i.e.
    distribute the fwd_proj_measurements according to the kn_distribution but not blur the measurements
    Input:
        fwd_proj_measurements: the forward proj measurements for each incident angle
        delta_ang: sampling angle in polar direction
    Output:
        new_photon_data: the compton scattered values [i, 0] for each angles [i, 1] and [i, 2]"""
    kn_dist = klein_nishina_correct_alpha(epsilon, delta_ang)
    n_photons = fwd_proj_measurements
    global_coor_list = []
    global_values_list = []

    for i in range(n_photons.shape[0]):
        photons, t, p = n_photons[i, :]
        assert (0 <= t <= np.pi)
        assert (0 <= p <= 2 * np.pi)
        photon_dist_alpha = photons * kn_dist[:, 0]  # get the compton scattered values over alpha

        photon_dist_list = []  # want this to have shape (m, 3) with 0 #photons, 1 alpha, 2 beta
        for j in range(photon_dist_alpha.shape[0]):
            # for each alpha value, get the distribution over beta
            beta_dist = kn_distribution(photon_dist_alpha[j], kn_dist[j, 1], delta_ang)
            for k in range(len(beta_dist)):
                photon_dist_list.append([beta_dist[k, 0], kn_dist[j, 1], beta_dist[k, 1]])
                # print(photon_dist_list[-1], 'photon list dist')

        photon_dist_list = np.asarray(photon_dist_list)
        # now I have the compton scattered values in a list
        coordinates = []
        values = []
        for i in range(len(photon_dist_list)):
            photon, alpha, beta = photon_dist_list[i, :]
            coor = [np.sin(alpha)*np.cos(beta), np.sin(alpha)*np.sin(beta), np.cos(alpha)]
            coordinates.append(coor)
            values.append(photon)

        # do rotation
        # perform some reshaping of the arrays to match the format expected by matrix rotation
        coordinates = np.asarray(coordinates)
        values = np.asarray(values)
        coordinates = np.reshape(coordinates, (-1, 3))
        values = np.reshape(values, (-1))

        # rotate all values to the correct incident angle t, p
        new_coordinates = rotation_on_unit_sphere(np.transpose(coordinates), t, p)
        norms = np.linalg.norm(new_coordinates, axis=0)
        for n in norms:
            assert abs(n - 1) < 10 ** (-10)
        # add new coordinates and values to global list
        global_values_list.append(values)
        global_coor_list.append(np.transpose(new_coordinates))

    global_values_list = np.asarray(global_values_list)
    global_coor_list = np.asarray(global_coor_list)
    # reshape global lists
    global_values_list = np.reshape(global_values_list, (-1))
    global_coor_list = np.reshape(global_coor_list, (-1, 3))

    # return to spherical coordinates
    new_angles = np.asarray([cartesian_to_spherical(global_coor_list[i, :]) for i in range(global_coor_list.shape[0])])

    new_photon_data = np.column_stack((global_values_list, new_angles))

    return new_photon_data


def snr(values_d, values_bg, da):
    """Calculate the SNR using the projection values
        values_d: delta_Lambda (see report for what Lambda is)
        values_bg: Lambda
        da: delta angle, the sampling angle we use"""
    for i in range(values_bg.shape[0]):
        assert values_bg[i, 1] == values_d[i, 1]
        assert values_bg[i, 2] == values_d[i, 2]

    d2 = np.multiply(values_d[:, 0], values_d[:, 0])
    with np.errstate(divide='warn', invalid='ignore'):  # raise warning for float/0 but not for 0/0
        d2 = np.divide(d2, values_bg[:, 0])
        d2[np.isnan(d2)] = 0
    d2 = np.multiply(d2, np.sin(values_d[:, 1]) * da ** 2)  # should maybe multiply with the sampling angle
                                                            # for the compton distribution
    d2 = np.sum(d2)
    return d2


def plot_detectability(kappas):
    """Function for plotting the SNR values for different kappas divided by the SNR for the perfect detector
    Input: kappas, list of kappa values for the von Mises Fisher distribution
    """
    gamma = 0
    zd = 0
    delta_ang = 0.05
    files_vd = [f'vd_kappa{k}_da{delta_ang}.npy' for k in kappas]
    files_bg = [f'vbg_kappa{k}_da{delta_ang}.npy' for k in kappas]

    d2 = []
    for i in range(len(kappas)):
        values_d = np.load(files_vd[i])
        values_bg = np.load(files_bg[i])
        mask_d = values_d[:, 0] != 0
        values_d = values_d[mask_d]
        mask_bg = values_bg[:, 0] != 0
        values_bg = values_bg[mask_bg]
        d2.append(snr(values_d, values_bg, delta_ang))

    perf_d = np.load('fwd_d_da0.05.npy')
    perf_bg = np.load('fwd_bg_da0.05.npy')
    mask = perf_bg[:, 0] != 0
    mask2 = perf_d[:, 0] != 0
    perf_bg = perf_bg[mask]
    perf_d = perf_d[mask2]

    perf_d = perfect_detector(perf_d, delta_ang)
    perf_bg = perfect_detector(perf_bg, delta_ang)
    d2_perf = snr(perf_d, perf_bg, delta_ang)

    comparison = d2/d2_perf
    save = True
    if save:
        kappas.append(np.inf)
        d2.append(d2_perf)
        arr = np.stack((kappas, d2), axis=1)
        np.save(f'detectability_deltaang{delta_ang}_comptonperf.npy', arr)
        kappas.pop(-1)

    plt.figure()
    plt.semilogx(kappas, comparison)
    plt.xlabel(r'$\kappa$', fontsize=14)
    plt.ylabel(r'$\frac{SNR(\kappa)}{SNR_{perfect}}$', fontsize=14)
    plt.show()



def debug_code():
    """Function for debugging the code, using data from simulations to calculate the SNR to see what happens.
    It is a bit messy. """
    #plot_detectability([0.001, 0.01, 0.1, 1, 10, 100])
    #detectability_first_step(0, 0, 0.05)
    #detectability_kappa(0, 0, 0.05, [10])
    test = True
    if test:

        delta_ang = 0.05
        # load the forward projection for both cases and remove the zeros to save memory
        n_photons_delta = np.load(f'fwd_d_da{delta_ang}.npy')
        n_photons_bg = np.load(f'fwd_bg_da{delta_ang}.npy')
        #print(len(n_photons_delta), n_photons_bg.shape)
        mask = n_photons_delta[:, 0] != 0
        n_photons_delta = n_photons_delta[mask]
        mask = n_photons_bg[:, 0] != 0
        n_photons_bg = n_photons_bg[mask]
        #print(len(n_photons_delta), len(n_photons_bg))

        # calculate the perfect values of Lambda(theta, phi)
        # setting perf_values_i = n_photons_i and then do the calculations is the perfect detector
        perf_values_d = perfect_detector(n_photons_delta, delta_ang)
        perf_values_bg = perfect_detector(n_photons_bg, delta_ang)
        #print(perf_values_d.shape, perf_values_bg.shape)
        for i in range(perf_values_bg.shape[0]):
            assert perf_values_bg[i, 1] == perf_values_d[i, 1]
            assert perf_values_bg[i, 2] == perf_values_bg[i, 2]

        # calculate the snr for the perfect case
        d2_perf = np.multiply(perf_values_d[:, 0], perf_values_d[:, 0])
        with np.errstate(divide='warn', invalid='ignore'):  # raise warning for float/0 but not for 0/0
            d2_perf = np.divide(d2_perf, perf_values_bg[:, 0])
            d2_perf[np.isnan(d2_perf)] = 0

        d2_perf = np.multiply(d2_perf, np.sin(perf_values_d[:, 1]) * delta_ang ** 2)
        d2_perf_sum = np.sum(d2_perf)
        print(d2_perf_sum)  # this is the snr value

        # calculate the snr value for a specific kappa value
        values_d = np.load('vd_kappa10_da0.05.npy')
        values_bg = np.load('vbg_kappa10_da0.05.npy')
        mask = values_d[:, 0] != 0
        values_d = values_d[mask]
        mask = values_bg[:, 0] != 0
        values_bg = values_bg[mask]
        #print(values_d.shape, values_bg.shape)
        #for i in range(values_bg.shape[0]):
        #    assert values_bg[i, 1] == values_d[i, 1]
        #    assert values_bg[i, 2] == values_d[i, 2]

        d2_2 = np.multiply(values_d[:, 0], values_d[:, 0])
        # with np.errstate(divide='warn', invalid='ignore'):  # raise warning for float/0 but not for 0/0
        d2_2 = np.divide(d2_2, values_bg[:, 0])
        #    d2_perf[np.isnan(d2_perf)] = 0
        d2_2 = np.multiply(d2_2, np.sin(values_d[:, 1]) * delta_ang ** 2)
        d2_sum = np.sum(d2_2)
        print(d2_sum, 'snr not perfect')
        print(d2_sum/d2_perf_sum)

        # trying to make the measurements into bins that can then be integrated over to see if this influences
        # the snr value significantly
        arr = values_bg
        bin_sums_bg = return_bins(arr)
        bin_sums_d = return_bins(values_d)

        d = []
        for i in range(bin_sums_d.shape[0]):
            t = delta_ang * i
            for j in range(bin_sums_d.shape[1]):
                with np.errstate(divide='warn', invalid='ignore'):
                    d.append(bin_sums_d[i, j]**2/bin_sums_bg[i, j] * np.sin(t) * delta_ang**2)

        d = np.nan_to_num(d)
        d = np.sum(d)
        print(d, 'd')
        print(d/d2_perf_sum, 'd2/d2_perf')


def return_bins(arr):
    """Return the values of the arrays (Lambda(theta, phi, alpha) but binned before calculating the snr """
    delta_ang = 0.1
    values = arr[:, 0]
    theta = arr[:, 1]
    phi = arr[:, 2]

    num_bins_theta = int(np.pi/delta_ang) +1  # Number of bins for theta
    num_bins_phi = int(2*np.pi/delta_ang) + 1 # Number of bins for phi

    theta_bins = np.linspace(min(theta), max(theta), num_bins_theta)
    phi_bins = np.linspace(min(phi), max(phi), num_bins_phi)

    theta_bin_indices = np.digitize(theta, theta_bins)
    phi_bin_indices = np.digitize(phi, phi_bins)
    bin_sums = np.zeros((num_bins_theta, num_bins_phi))
    for i in range(len(values)):
        theta_bin_index = theta_bin_indices[i] - 1  # Subtract 1 to match 0-based indexing
        phi_bin_index = phi_bin_indices[i] - 1

        bin_sums[theta_bin_index, phi_bin_index] += values[i]

    return bin_sums


def plot_sd():
    # plot source distribution and detector to have in report

    # Create an array of angles
    angles = np.linspace(0, 2 * np.pi, 300)

    # Calculate x and y coordinates for each circle
    x1 = Rs * np.cos(angles)
    y1 = Rs * np.sin(angles)

    x2 = R * np.cos(angles)
    y2 = R * np.sin(angles)

    x3 = Rd * np.cos(angles)
    y3 = Rd * np.sin(angles)

    # Create the plot
    plt.figure(figsize=(6, 6))
    plt.plot(x1, y1, label='Lesion', color='r')
    plt.plot(x2, y2, label='Head', color='b')
    plt.plot(x3, y3, label='Detector', color='k')
    plt.xlabel('Position [cm]')
    plt.ylabel('Position [cm]')

    # Set aspect ratio to 'equal' to ensure circles are circular
    #plt.gca().set_aspect('equal', adjustable='box')

    # Add legend
    plt.legend()

    # Show the plot
    plt.show()


def sphere_picture():
    """Plot picture of sphere and vector to have in report"""
    # Create a figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Define the radius and angle for the half sphere
    radius = 1.0
    theta = np.linspace(0, np.pi/2, 100)
    phi = np.linspace(0, 2 * np.pi, 100)

    # Create a meshgrid for the spherical coordinates
    theta, phi = np.meshgrid(theta, phi)

    # Calculate the x, y, and z coordinates for the half sphere
    x = radius * np.sin(theta) * np.cos(phi)
    y = radius * np.sin(theta) * np.sin(phi)
    z = radius * np.cos(theta)

    # Plot the half sphere
    ax.plot_surface(x, y, z, color='b', alpha=0.2)

    # Define the position of the vector
    vector_position = (0, 0, 0)
    p = np.pi/4
    t = np.pi/4
    vec = [radius*np.cos(p)*np.sin(t), radius*np.sin(t)*np.sin(p), radius*np.cos(t)]

    # Plot the vector
    ax.quiver(*vector_position, vec[0], vec[1], vec[2], color='k') #, label='Vector')

    # Set axis labels
    ax.set_xlabel("X'")
    ax.set_ylabel("Y'")
    ax.set_zlabel("Z'")

    # Annotate the vector
    ax.text(vec[0], vec[1], vec[2], r'$(\theta, \phi)$', color='k')

    # Add a legend
    #ax.legend()

    # Set the aspect ratio to be equal
    ax.set_box_aspect([1, 1, 0.5])

    # Show the plot
    plt.show()


def main():
    """Basic simulations are done through
    1. run dectectability_first_step to get the fwd proj measurements for that detector pixel
    2. run detectability_kappa to get the lambda values for each kappa (using the saved data from 1)
    3. perform calculations of the snr like in debug_code for the non-perfect and perfect detector using the data from 2"""
    #detectability_kappa(0, 0, 0.25, [1, 2, 5, 10, 15])
    #debug_code()
    #plot_sd()
    sphere_picture()


if __name__ == '__main__':
    main()