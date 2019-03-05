import os

# Third-party
from astropy.io import fits
from astropy.time import Time
import astropy.units as u
import h5py
import numpy as np
from schwimmbad import choose_pool

from thejoker import TheJoker, RVData, JokerParams
from thejoker.log import log as joker_logger


def get_stars_visits(allstar_file, allvisit_file, apogee_ids=None,
                     min_nvisits=3):
    allstars = fits.getdata(os.path.abspath(allstar_file))
    allvisits = fits.getdata(os.path.abspath(allvisit_file))

    if apogee_ids is not None:
        allstars = allstars[np.isin(allstars['APOGEE_ID'], apogee_ids)]
        allvisits = allvisits[np.isin(allvisits['APOGEE_ID'], apogee_ids)]

    visits_mask = (np.isfinite(allvisits['VHELIO']) &
                   np.isfinite(allvisits['VRELERR']) &
                   (np.abs(allvisits['VHELIO']) < 1000.) & # MAGIC NUMBER
                   (np.abs(allvisits['VRELERR']) < 100.)) # MAGIC NUMBER

    # LOW_SNR, PERSIST_HIGH, PERSIST_JUMP_POS, PERSIST_JUMP_NEG
    skip_mask = np.sum(2 ** np.array([4, 9, 12, 13]))
    visits_mask &= (allvisits['STARFLAG'] & skip_mask) == 0

    # VERY_BRIGHT_NEIGHBOR, SUSPECT_RV_COMBINATION, SUSPECT_BROAD_LINES
    skip_mask = np.sum(2 ** np.array([3, 16, 17]))
    visits_mask &= (allvisits['STARFLAG'] & skip_mask) == 0

    v_apogee_ids, counts = np.unique(allvisits['APOGEE_ID'][visits_mask],
                                     return_counts=True)

    stars_mask = np.isin(allstars['APOGEE_ID'],
                         v_apogee_ids[counts >= min_nvisits])
    stars = allstars[stars_mask]
    stars = stars[np.unique(stars['APOGEE_ID'], return_index=True)[1]]

    visits = allvisits[visits_mask]
    visits = visits[np.isin(visits['APOGEE_ID'], np.unique(stars['APOGEE_ID']))]

    return stars, visits


def main(pool):
    # ------------------------------------------------------------------------
    # Configuration:
    min_nvisits = 6
    requested_n_samples = 128
    # prior_samples_file = '/mnt/ceph/users/apricewhelan/projects/hq/cache/P1-32768_prior_samples.hdf5'
    prior_samples_file = os.path.expanduser('~/.hq/test_prior_samples.hdf5')
    # ------------------------------------------------------------------------

    # Load all data:
    stars_r11, visits_r11 = get_stars_visits('../data/allField-r11.fits',
                                             '../data/allFieldVisits-r11.fits',
                                             min_nvisits=min_nvisits)
    stars_r11 = stars_r11[stars_r11['RV_LOGG'] <= 3.4]

    stars_r10, visits_r10 = get_stars_visits('../data/allStar-r10-l31c-58297.fits',
                                             '../data/allVisit-r10-l31c-58297.fits',
                                             apogee_ids=stars_r11['APOGEE_ID'],
                                             min_nvisits=min_nvisits)

    # Filter down to match source and visits exactly:
    r11_mask = np.isin(stars_r11['APOGEE_ID'],
                       stars_r10['APOGEE_ID'])

    stars_r11 = stars_r11[r11_mask]
    visits_r11 = visits_r11[np.isin(visits_r11['APOGEE_ID'],
                                    stars_r11['APOGEE_ID'])]

    r10_mask = np.isin(np.array([x[12:] for x in visits_r10['FILE']]),
                       np.array([x[12:] for x in visits_r11['FILE']]))
    visits_r10 = visits_r10[r10_mask]

    r11_mask = np.isin(np.array([x[12:] for x in visits_r11['FILE']]),
                       np.array([x[12:] for x in visits_r10['FILE']]))
    visits_r11 = visits_r11[r11_mask]

    v10_apogee_ids, counts = np.unique(visits_r10['APOGEE_ID'],
                                       return_counts=True)
    stars_r10 = stars_r10[np.isin(stars_r10['APOGEE_ID'],
                                  v10_apogee_ids[counts >= min_nvisits])]

    v11_apogee_ids, counts = np.unique(visits_r11['APOGEE_ID'],
                                       return_counts=True)
    stars_r11 = stars_r11[np.isin(stars_r11['APOGEE_ID'],
                                  v11_apogee_ids[counts >= min_nvisits])]

    assert len(stars_r10) == len(stars_r11)
    assert len(visits_r10) == len(visits_r11)

    # Prepare a file to cache everything to:
    os.makedirs('cache', exist_ok=True)

    # Ensure cache files for samples exist:
    for filename in ['cache/r10-samples.hdf5', 'cache/r11-samples.hdf5']:
        if not os.path.exists(filename):
            with h5py.File(filename, 'w') as f: # create file
                pass

    cache_filenames = ['cache/r10-samples.hdf5',
                       'cache/r11-samples.hdf5']

    run_names = ['r10', 'r11']

    all_visits = [visits_r10, visits_r11]

    # TODO: customize settings!
    params = JokerParams(P_min=2*u.day, P_max=32768.*u.day,
                         jitter=(np.log(1000**2), 4),
                         jitter_unit=u.m/u.s,
                         poly_trend=2)
    joker = TheJoker(params, pool=pool)

    # Loop over stars and generate posterior samplings:
    for apogee_id in stars_r11['APOGEE_ID']:
        for cache_filename, run_name, visits in zip(cache_filenames,
                                                    run_names,
                                                    all_visits):
            visits = visits[visits['APOGEE_ID'] == apogee_id]

            if run_name == 'r10':
                print('{0}: {1} visits'.format(apogee_id, len(visits)))

            with h5py.File(cache_filename, 'r') as f:
                if apogee_id in f:
                    print('Star {0} already done'.format(apogee_id))
                    continue

            data = RVData(t=Time(visits['JD'], format='jd', scale='tcb'),
                          rv=visits['VHELIO'] * u.km/u.s,
                          stddev=visits['VRELERR'] * u.km/u.s)

            try:
                samples, ln_prior, ln_like = joker.iterative_rejection_sample(
                    data=data, n_requested_samples=requested_n_samples,
                    prior_cache_file=prior_samples_file,
                    return_logprobs=True)

            except Exception as e:
                print("\t Failed sampling for star {0} \n Error: {1}"
                      .format(apogee_id, str(e)))
                continue

            with h5py.File(cache_filename, 'r+') as f:
                g = f.create_group(apogee_id)
                samples.to_hdf5(g)

                g.create_dataset('ln_prior', data=ln_prior)
                g.create_dataset('ln_like', data=ln_like)


if __name__ == "__main__":
    from argparse import ArgumentParser
    import logging

    # Define parser object
    parser = ArgumentParser(description="")

    vq_group = parser.add_mutually_exclusive_group()
    vq_group.add_argument('-v', '--verbose', action='count', default=0,
                          dest='verbosity')
    vq_group.add_argument('-q', '--quiet', action='count', default=0,
                          dest='quietness')

    # parser.add_argument("--overwrite", dest="overwrite", default=False,
    #                     action="store_true",
    #                     help="Overwrite any existing results for this "
    #                          "JokerRun.")

    group = parser.add_mutually_exclusive_group()
    group.add_argument("--procs", dest="n_procs", default=1,
                       type=int, help="Number of processes.")
    group.add_argument("--mpi", dest="mpi", default=False,
                       action="store_true", help="Run with MPI.")

    args = parser.parse_args()

    # Set logger level based on verbose flags
    if args.verbosity != 0:
        if args.verbosity == 1:
            joker_logger.setLevel(logging.DEBUG)
        else: # anything >= 2
            joker_logger.setLevel(1)

    elif args.quietness != 0:
        if args.quietness == 1:
            joker_logger.setLevel(logging.WARNING)
        else: # anything >= 2
            joker_logger.setLevel(logging.ERROR)

    else: # default
        joker_logger.setLevel(logging.INFO)

    pool_kwargs = dict(mpi=args.mpi, processes=args.n_procs)
    pool = choose_pool(**pool_kwargs)

    main(pool=pool)
