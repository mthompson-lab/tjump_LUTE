#! /usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from h5py import File
from pathlib import Path
from scipy.signal import medfilt
from scipy.stats import linregress, zscore
from scipy.interpolate import CubicSpline, splrep, BSpline
from sys import argv
import pandas as pd
from scipy.optimize import curve_fit
from lmfit.models import GaussianModel, VoigtModel
from scipy import sparse
from scipy.sparse.linalg import spsolve

def eV_to_lamba(eV):
    "E (eV) = 1239.8 / l (nm)"
    lambda_nm = 1239.8/eV
    lambda_ang = lambda_nm*10
    return lambda_ang

def two_theta_deg_to_q(two_theta, wavelength):
    q = (4*np.pi/wavelength)*np.sin((two_theta*np.pi/180)/2)
    return q

def temp_calc(data_in, regression_in):
    m = regression_in[0]
    x = data_in
    b = regression_in[1]
    y = m*x+b
    return y

def simple_peak_fit(x_in,y_in, temp):
    fit_mask = np.array(x_in, dtype=bool)
    fit_mask[x_in<1.6]=False
    fit_mask[x_in>2.3]=False
    tck_s = splrep(x_in[fit_mask], y_in[fit_mask], s=len(x_in[fit_mask]))
    fit = BSpline(*tck_s)(x_in)[fit_mask]
    idx = fit.argmax()
    q_val = x_in[fit_mask][idx]
    # plt.scatter(temp,q_val)
    return q_val


def baseline_als(y, lam=1e6, p=0.01, niter=10):
    """Asymmetric Least Squares baseline subtraction."""
    L = len(y)
    D = sparse.diags([1, -2, 1], [0, -1, -2], shape=(L, L - 2))
    w = np.ones(L)
    for _ in range(niter):
        W = sparse.spdiags(w, 0, L, L)
        Z = W + lam * D @ D.T
        z = spsolve(Z, w * y)
        w = p * (y > z) + (1 - p) * (y < z)
    return z

def two_peak_fit(x_in, y_in, temp=0, plot=False, verbose=False, model_type='gaussian', show_derivatives=False):
    """
    Fit two peaks to spectrum using Gaussian or Voigt models.
    
    Parameters:
        x_in, y_in (array-like): Input data
        temp (float): Optional temperature label for plots
        plot (bool): Show fitted curve and residuals
        verbose (bool): Print full fit report
        model_type (str): 'gaussian' or 'voigt'
        show_derivatives (bool): Show 1st and 2nd derivatives for diagnostics
    
    Returns:
        delta_q (float): Difference between peak centers
    """
    # Mask and crop region of interest
    fit_mask = (x_in > 1.0) & (x_in < 3.8)
    x = np.array(x_in)[fit_mask]
    y = np.array(y_in)[fit_mask]

    # Baseline subtraction
    baseline = baseline_als(y)
    y_corr = y - baseline

    # Show derivatives (diagnostic only)
    if show_derivatives:
        dy = np.gradient(y_corr, x)
        d2y = np.gradient(dy, x)
        plt.plot(x, y_corr, label="Corrected")
        plt.plot(x, dy, '--', label="1st Derivative")
        plt.plot(x, d2y, ':', label="2nd Derivative")
        plt.legend(); plt.title("Derivative Check"); plt.grid(True); # plt.show()
        plt.savefig("{}/run{:04d}_Derivative_Check.png".format(output_dir,run), dpi=100)

    # Choose model type
    if model_type.lower() == 'voigt':
        g1 = VoigtModel(prefix='g1_')
        g2 = VoigtModel(prefix='g2_')
    else:
        g1 = GaussianModel(prefix='g1_')
        g2 = GaussianModel(prefix='g2_')

    model = g1 + g2
    params = model.make_params()

    # Initial guesses and bounds
    params['g1_amplitude'].set(value=max(y_corr), min=0)
    params['g1_center'].set(value=1.9, min=1.5, max=2.1)
    params['g1_sigma'].set(value=0.2, min=0.05, max=0.6)

    params['g2_amplitude'].set(value=max(y_corr)/2, min=0)
    params['g2_center'].set(value=2.7, min=2.3, max=3.1)
    params['g2_sigma'].set(value=0.2, min=0.05, max=0.6)

    # Additional Voigt-specific params (gamma)
    if model_type.lower() == 'voigt':
        params['g1_gamma'].set(value=0.1, min=0.01)
        params['g2_gamma'].set(value=0.1, min=0.01)

    # Perform fit
    result = model.fit(y_corr, params, x=x)

    # Report
    if verbose:
        print(result.fit_report())

    if plot:
        result.plot_fit()
        plt.grid(True)
        plt.title(f"{model_type.title()} Two-Peak Fit @ Temp {temp}K")
        plt.savefig("{}/run{:04d}_Two-Peak_Fit_Temp_{}K.png".format(output_dir,run,temp), dpi=100)

        comps = result.eval_components(x=x)
        plt.plot(x, comps['g1_'], label='Peak 1')
        plt.plot(x, comps['g2_'], label='Peak 2')
        plt.legend()
        plt.grid(True)
        plt.title("Individual Peak Components")
        plt.savefig("{}/run{:04d}_Individual_Peak_Components.png".format(output_dir,run), dpi=100)

    # Extract peak positions and compute delta
    c1 = result.params['g1_center'].value
    c2 = result.params['g2_center'].value
    amp2 = result.params['g2_amplitude'].value

    # Reliability warning
    if amp2 < 0.05 * max(y_corr):
        print("Warning: Second peak may be unreliable (low amplitude).")

    return c2 - c1

def exponential_decay(x, A, k, C, D):
    """
    Exponential decay function.
    A: Initial amplitude
    k: Decay rate
    C: Offset/asymptote
    """
    return A * np.exp(-k * (x-D)) + C



def rescale(q, int_arr, method='integration'):
    rescaled_int_arr_list = []
    refy = int_arr[0]
    scaling_mask = np.array(q, dtype=bool)
    scaling_mask[q<1.0]=False # 1 lower and 1.96 upper puts even at isosbestic point of 1.48, seems to break t-jump signal though
    scaling_mask[q>3.0]=False
    for y in int_arr:
        yy = y[scaling_mask]
        refyy = refy[scaling_mask]
        xx = q[scaling_mask]
        if method == 'algebraic':
            q_ref = xx*refyy
            q_sig = xx*yy
            top = np.dot(q_sig,q_ref)
            bottom = np.dot(q_sig,q_sig)
            scalar = top/bottom
        elif method == 'integration':
            scalar = refyy.sum()/yy.sum()
        else:
            raise TypeError('please choose from the following scaling approaches: algebraic, integration')
        rescaled = y*scalar
        rescaled_int_arr_list.append(rescaled)
    out = np.array(rescaled_int_arr_list)
    return out

### idea: make a filter which will remove any curve with an outlier at any q...save only those with all entries below z = ??
def z_filt(array, z_cutoff):
    from scipy import stats
    z = stats.zscore(array, axis=0)
    zabs = np.abs(z)
    filt = []
    zcut = zabs>z_cutoff
    for row in zcut:
        if row.any():
            filt.append(False)
        else:
            filt.append(True)
    return np.array(filt)




###------------------------------------------------------------------------------------------

#params to modify
# run = 146
# exp = 'mfx101080524'


### -------------------------------------------------------------------------------------------------------------------------


def spline_peak_fit(x_in, y_in):
    from scipy.signal import savgol_filter
    fit_mask = (x_in >= 1.6) & (x_in <= 2.3)
    smoothed = savgol_filter(y_in[fit_mask], 11, 3)
    idx = np.argmax(smoothed)
    return x_in[fit_mask][idx]

def fit_peak(q, curve, temp=0, plot=False, verbose=False, model_type='gaussian', show_derivatives=False, method="two_peak_fit"):
    if method == "two_peak_fit":
        return two_peak_fit(q, curve, temp=temp, plot=plot, verbose=verbose, model_type=model_type, show_derivatives=show_derivatives)
    elif method == "simple":
        return simple_peak_fit(q, curve, temp)
    elif method == "spline":
        return spline_peak_fit(q, curve)
    else:
        raise ValueError(f"Unknown peak fitting method: {method}")
    
def main():

    ### I/O related arguments
    import argparse
    parser=argparse.ArgumentParser(
        description='''analysis of LCLS smalldata HDF5 files on a per run basis - compute t-jump by fitting water peak''',
        epilog=""" """)
    parser.add_argument('--exp', 
                        type=str, 
                        required=True, 
                        help="str (eg: 'mfx101080524') - LCLS experiment identifier")
    
    parser.add_argument('--run', 
                        type=int, 
                        required=True, 
                        help="int (eg: 146) - LCLS run number")
    
    parser.add_argument('--output', 
                        type=str, 
                        default='', 
                        help="str (eg: '/path/to/') - destination for the folder containing output PNG and HDF5 files, "
                             "default is /sdf/data/lcls/ds/{exp[:3]}/{exp}/stats/summary/TJump")
    
    parser.add_argument('--output_h5', 
                        type=str, 
                        default='', 
                        help="str (eg: 'run0005_sd2qwp1.hdf5') - name of the output h5 file, "
                             "default is run{:04d}_smalldata_to_Tval_multimethod.hdf5")
    
    parser.add_argument('--input', 
                        type=str, 
                        default='', 
                        help="str (eg: '/path/to/') - path to input smalldata h5 file, "
                             "default is /sdf/data/lcls/ds/{exp[:3]}/{exp}/hdf5/smalldata/{exp}_Run{:04}.h5")
    
    parser.add_argument('--peakfit', 
                        type=str, 
                        default='simple', 
                        help="Peak fitting method: 'simple' or 'spline' or 'two_peak_fit' (default=simple)")
    
    args=parser.parse_args()
    run = args.run
    exp = args.exp
    if not args.output:
        output_dir = "/sdf/data/lcls/ds/{}/{}/stats/summary/TJump".format(exp[:3],exp)
        print("no output directory specified, defaulting to:\n{}".format(output_dir))
    else:
        output_dir = args.output
    if not args.output_h5:
        output_h5 = "run{:04d}_sd2qwp1.hdf5".format(run)
    else:
        output_h5 = args.output_h5
    if not args.input:
        f = '/sdf/data/lcls/ds/{}/{}/hdf5/smalldata/{}_Run{:04}.h5'.format(exp[:3],exp,exp,run)
    else:
        f = args.input
    if args.peakfit not in ['simple', 'spline', 'two_peak_fit']:
        raise ValueError(f"Peak fitting method must be in ['simple', 'spline', 'two_peak_fit'], got {args.peakfit}")
    peakfit_method = args.peakfit

    h5 = File(f) ### using h5py
    #logfile = open("run{}.logfile".format(run), 'w')
    xray_on = h5['lightStatus']['xray'][:] == 1
    count_xray_on = xray_on[xray_on == True].shape[0]
    print("{} events with X-rays detected\n".format(count_xray_on))

    ### use average wavelength from run as shot-to-shot variation contributes less than z-fluctuations
    eV = np.nanmean(h5['ebeam']['photon_energy'][xray_on][:])
    lamb = eV_to_lamba(eV)
    two_theta = h5['Rayonix']['pyfai_q'][xray_on][0] ### two_theta is the same for all events
    q = two_theta_deg_to_q(two_theta, lamb)

    try:
        lens_v = h5['scan']['lens_v'][:]
        print('lens_v scan')
    except:
        print('not a lens_v scan')
    try:
        lens_h = h5['scan']['lens_h'][:]
        print('lens_h scan')
    except:
        print('not a lens_h scan')


    ### scale using total intensity from beammon and then drop files that don't have appropriate falloff...shutter closed or noisy files etc
    beam_scale_factor = h5['ipm_dg2']['sum'][xray_on][:]
    beam_scale_factor_mask = np.array(beam_scale_factor, dtype=bool)
    beam_scale_factor_sigma = np.abs(zscore(beam_scale_factor))
    beam_scale_factor_mask[beam_scale_factor_sigma>2.0]=False

    azav_all = h5['Rayonix']['pyfai_azav'][:]
    if len(azav_all.shape) == 3:
        print('2d azimuthal averaging')
        azav_all = np.average(azav_all, axis=1)
    elif len(azav_all.shape) == 2:
        print('1d azimuthal averaging')
    else:
        print("azav shape error")

    azav = azav_all[xray_on]

    scaled_azav = azav[beam_scale_factor_mask] / np.vstack(beam_scale_factor[beam_scale_factor_mask])
    #scaled_azav = scaled_azav_0[beam_scale_factor_mask]

    junk_low_q_target = 2.0
    junk_low_q_diff = np.absolute(q-junk_low_q_target)
    junk_low_q_index = junk_low_q_diff.argmin()

    junk_high_q_target = 3.0
    junk_high_q_diff = np.absolute(q-junk_high_q_target)
    junk_high_q_index = junk_high_q_diff.argmin()

    junk_filt = np.abs(scaled_azav[:,junk_high_q_index]) < np.abs(0.7*scaled_azav[:,junk_low_q_index])
    junk_filt[0] = False
    filt_scaled_azav = scaled_azav[junk_filt]

    count_reject_beam_scale_factor = beam_scale_factor_mask[beam_scale_factor_mask == False].shape[0]
    print("{} events rejected based on 2-sigma beam_scale_factor filter\n".format(count_reject_beam_scale_factor))
    count_reject_q3q2_ratio = junk_filt[junk_filt == False].shape[0]
    print("{} events rejected based on scattering falloff filter\n".format(count_reject_q3q2_ratio))


    ids = h5['event_time']
    ids_beam_scale_factor_mask = ids[xray_on][beam_scale_factor_mask]
    ids_q3q2_mask = ids_beam_scale_factor_mask[junk_filt]


    output = pd.DataFrame()
    output['event_time'] = h5['event_time']
    output['xray_on'] = h5['lightStatus']['xray'][:]
    output['beam_scale_factor_2sigma_filter_pass'] = output['event_time'].isin(ids_beam_scale_factor_mask)
    output['scattering_falloff_filter_pass'] = output['event_time'].isin(ids_q3q2_mask)

    ### define q_filter for plotting
    plot_mask = np.array(q, dtype=bool)
    plot_mask[q<0.3]=False
    plot_mask[q>3.2]=False

    fig, ax = plt.subplots()
    ax.axvline(1.93, 0, 1, color='black')
    ax.set_title("Scaled to BeamMon")
    ax.set_xlabel("q (1/Å)")
    ax.set_ylabel("I (AU)")
    for c in filt_scaled_azav:
        ax.plot(q[plot_mask], c[plot_mask])
    fig.savefig("{}/run{:04d}_scattering_falloff_filter_pass.png".format(output_dir,run), dpi=200)


    ### remove Bragg peak contributions to azav
    smooth_filt_scaled_azav = medfilt(filt_scaled_azav[:,:], 13)

    ### remove air-scatter to best approximation
    subtract = smooth_filt_scaled_azav.min(axis=0)
    test = smooth_filt_scaled_azav - subtract

    fig2, ax2 = plt.subplots()
    ax2.axvline(1.93, 0, 1, color='black')
    ax2.set_title("Scaled, Smoothed, AirSubtracted")
    ax2.set_xlabel("q (1/Å)")
    ax2.set_ylabel("I (AU)")
    for c in test:
        ax2.plot(q[plot_mask], c[plot_mask])

    fig2.savefig("{}/run{:04d}_Scaled_Smoothed_AirSubtracted.png".format(output_dir,run), dpi=200)


    ### drop curves that have insignificant solvent contribution (keep "hits" only)
    water_low_q_target = 1.0
    water_low_q_diff = np.absolute(q-water_low_q_target)
    water_low_q_index = water_low_q_diff.argmin()
    water_high_q_target = 1.93
    water_high_q_diff = np.absolute(q-water_high_q_target)
    water_high_q_index = water_high_q_diff.argmin()
    factor = 2.5
    water_filt = np.abs(test[:,water_high_q_index]) > np.abs(factor*test[:,water_low_q_index])
    test_filt = test[water_filt]

    fig3, ax3 = plt.subplots()
    ax3.axvline(1.93, 0, 1, color='black')
    ax3.set_title("Scaled, Smoothed, AirSubtracted, HighWater")
    ax3.set_xlabel("q (1/Å)")
    ax3.set_ylabel("I (AU)")
    for c in test_filt:
        ax3.plot(q[plot_mask], c[plot_mask])
    fig3.savefig("{}/run{:04d}_water-to-air_filter_pass.png".format(output_dir,run), dpi=200)

    ids_waterq2q1ratio_mask = ids_q3q2_mask[water_filt]
    output['water-to-air_filter_pass'] = output['event_time'].isin(ids_waterq2q1ratio_mask)


    count_reject_airwater_ratio = water_filt[water_filt == False].shape[0]
    print("{} events rejected based on water-to-air ratio filter\n".format(count_reject_airwater_ratio))

    ### drop curves by water peak z-score
    water_peak_mean = test_filt[:,water_high_q_index].mean()
    water_peak_std = test_filt[:,water_high_q_index].std()
    water_peak_sigma = np.abs(test_filt[:,water_high_q_index] - water_peak_mean) / water_peak_std
    water_peak_filt =  np.array(water_peak_sigma, dtype=bool)
    water_peak_filt[water_peak_sigma>2.0] = False

    test_doublefilt = test_filt[water_peak_filt]
    # water_peak_vals = test_doublefilt[:,water_high_q_index]
    # test_doublefilt = test_doublefilt[water_peak_vals>0.0015]
    #water_filt = water_filt[water_peak_vals>0.0015]
    #water_peak_filt = water_peak_filt[water_peak_vals>0.0015]

    fig4, ax4 = plt.subplots()
    ax4.axvline(1.93, 0, 1, color='black')
    ax4.set_title("Scaled, Smoothed, AirSubtracted, HighWater-Peakfilt")
    ax4.set_xlabel("q (1/Å)")
    ax4.set_ylabel("I (AU)")
    for c in test_doublefilt:
        ax4.plot(q[plot_mask], c[plot_mask])
    fig4.savefig("{}/run{:04d}_water-peak-2sigma_filter_pass.png".format(output_dir,run), dpi=200)


    ids_q2_2sigma_mask = ids_waterq2q1ratio_mask[water_peak_filt]
    output['water-peak-2sigma_filter_pass'] = output['event_time'].isin(ids_q2_2sigma_mask)


    count_reject_water_peak_2sigma = water_peak_filt[water_peak_filt == False].shape[0]
    print("{} events rejected based on 2-sigma water peak filter\n".format(count_reject_water_peak_2sigma))


    ### setup standard curve of peak-position values as a function of temperature (switch axes for linear regression)
    ### hard-code paths to standard curve files
    Standard_vecs_x = np.load("/sdf/data/lcls/ds/mfx/mfx101080524/results/wolff/reference_vectors/singular_vectors_matching_q.npy")
    Standard_vecs_ys_buff = np.load("/sdf/data/lcls/ds/mfx/mfx101080524/results/wolff/reference_vectors/full_array_scaled_vecs.npy")[0:160]
    # Standard_vecs_ys_cell = np.load("/sdf/data/lcls/ds/mfx/mfx101080524/results/wolff/reference_vectors/full_array_scaled_vecs.npy")[160:320]
    temps = np.arange(280,341,4)
    buff_temp_ys = np.split(Standard_vecs_ys_buff, 16)
    lin_x = []
    lin_y = []
    for scatter,temp in zip(buff_temp_ys, temps):
        for curve in scatter:
            val = simple_peak_fit(Standard_vecs_x, curve, temp)
            val = fit_peak(Standard_vecs_x, curve, temp, method=peakfit_method)
            lin_y.append(temp)
            lin_x.append(val)
    #tempfinder = linregress(lin_x,lin_y)
    # p0 = [1, 0.5, 0, 280]
    A_init = max(lin_y) - min(lin_y)
    k_init = 1
    C_init = min(lin_y) 
    D_init = min(lin_x)
    p0 = [A_init, k_init, C_init, D_init]
    # Probably worth defining sensible bounds for the fit, and let exponential fit fail and fall back to linear 
    # regression when out of bounds
    fit_method = "exponential"
    try:
        # Perform the curve fit
        params, covariance = curve_fit(exponential_decay, lin_x, lin_y, p0=p0)
        # Check if the fit was successful
        if not np.all(np.isfinite(covariance)):
            raise RuntimeError("Covariance matrix contains non-finite values")
        # Extract the fitted parameters
        A_fit, k_fit, C_fit, D_fit = params
        # Generate the fitted curve
        x_fit = np.linspace(min(lin_x), max(lin_x), len(lin_y))
        y_fit = exponential_decay(x_fit, A_fit, k_fit, C_fit, D_fit)
        print(f"Exponential fit parameters: A={A_fit:.2f}, k={k_fit:.2f}, C={C_fit:.2f}, D={D_fit:.2f}")
    except RuntimeError as e:
        # also covers RuntimeError: Optimal parameters not found: Number of calls to function has reached maxfev = 1000.
        print(f"Exponential fit failed: {e}")
        print("Falling back to linear regression")
        fit_method = "linear"
        # Linear regression as fallback
        from scipy.stats import linregress
        tempfinder = linregress(lin_x, lin_y)
        slope, intercept, r_value, p_value, std_err = tempfinder
        x_fit = np.linspace(min(lin_x), max(lin_x), len(lin_y))
        y_fit = slope * x_fit + intercept
        print(f"Linear regression: slope={slope:.4f}, intercept={intercept:.2f}, r={r_value:.4f}")
    ss_tot = np.sum((lin_y - np.mean(lin_y))**2)
    ss_res = np.sum((lin_y - y_fit)**2)
    r_squared = 1 - (ss_res / ss_tot)
    print("R^2 ={}".format(r_squared))

    ### rescale data for varying water peak heights
    rescaled = rescale(q, test_doublefilt, method='integration')
    ### define q_filter for z-filter...not noisy outer portions of curves
    z_mask = np.array(q, dtype=bool)
    z_mask[q<1.0]=False
    z_mask[q>3.2]=False
    z_filt_for_rescaled = z_filt(rescaled[:,z_mask], 5)
    rescaled_filt = rescaled[z_filt_for_rescaled]

    count_reject_z = z_filt_for_rescaled [z_filt_for_rescaled  == False].shape[0]
    print("{} events rejected based on 5-sigma z filter\n".format(count_reject_z ))


    ### initiate laser filters, sized to match filtered arrays
    laser_on    = h5['evr']['code_203'][xray_on][beam_scale_factor_mask][:] == 1
    laser_off1  = h5['evr']['code_204'][xray_on][beam_scale_factor_mask][:] == 1
    laser_off2  = h5['evr']['code_205'][xray_on][beam_scale_factor_mask][:] == 1

    # laser_on_filt = laser_on[junk_filt][water_filt][water_peak_filt][water_peak_vals>0.0015][z_filt_for_rescaled]
    # laser_off1_filt = laser_off1[junk_filt][water_filt][water_peak_filt][water_peak_vals>0.0015][z_filt_for_rescaled]
    # laser_off2_filt = laser_off2[junk_filt][water_filt][water_peak_filt][water_peak_vals>0.0015][z_filt_for_rescaled]
    laser_on_filt = laser_on[junk_filt][water_filt][water_peak_filt][z_filt_for_rescaled]
    laser_off1_filt = laser_off1[junk_filt][water_filt][water_peak_filt][z_filt_for_rescaled]
    laser_off2_filt = laser_off2[junk_filt][water_filt][water_peak_filt][z_filt_for_rescaled]

    qWP_arr = np.array([fit_peak(q, curve, method=peakfit_method) for curve in rescaled_filt])

    qLOFF2_arr = np.array([fit_peak(q, curve, method=peakfit_method) for curve in rescaled_filt[laser_off2_filt]])
    qLOFF1_arr = np.array([fit_peak(q, curve, method=peakfit_method) for curve in rescaled_filt[laser_off1_filt]])
    qLON_arr = np.array([fit_peak(q, curve, method=peakfit_method) for curve in rescaled_filt[laser_on_filt]])


    plt.figure(figsize=(12,6),dpi=200)
    ylims = [min(lin_x)-0.01, max(lin_x)+0.01]
    plt.subplot(1,2,1)
    plt.title("Water Peak ({}) vs Temperature Fit ({})".format(peakfit_method, fit_method))
    plt.xlabel("Temperature (K)")
    plt.ylabel("Water Peak Q (1/Å)")
    plt.scatter(lin_y,lin_x, label='Original Data') # reverse axes due to fitting vs plotting
    plt.plot(y_fit, x_fit, color='red', label='R^2={}'.format(r_squared))
    plt.legend()
    plt.grid(True)
    plt.ylim(ylims)
    plt.subplot(1,2,2)
    plt.title("Water Peak ({}) vs Laser Status".format(peakfit_method))
    plt.violinplot([qLON_arr, qLOFF1_arr, qLOFF2_arr], positions=[0,1,2], showextrema=False, showmeans=True, showmedians=True)
    plt.xticks([0,1,2],["on","off1","off2"])
    plt.ylim(ylims)
    plt.savefig("{}/run{:04d}_water-peak-vs-laser-status.png".format(output_dir,run), dpi=200)

    ids_ZofI_5sigma_mask = ids_q2_2sigma_mask[z_filt_for_rescaled]
    output['zscore-I-5sigma_filter_pass'] = output['event_time'].isin(ids_ZofI_5sigma_mask)
    output['laser_on'] = h5['evr']['code_203'][:] == 1
    output['laser_off1'] = h5['evr']['code_204'][:] == 1
    output['laser_off2'] = h5['evr']['code_205'][:] == 1

    indices = output.loc[output['event_time'].isin(ids_ZofI_5sigma_mask)].index
    output['q-of-water-peak1'] = np.NaN
    output.loc[indices, 'q-of-water-peak1'] = qWP_arr

    output.to_hdf('{}/{}'.format(output_dir,output_h5), key='df', mode='w')
    return

if __name__ == "__main__":
    main()