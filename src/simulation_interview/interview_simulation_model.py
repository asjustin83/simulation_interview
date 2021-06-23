#!/usr/bin/env python
# coding: utf-8

import sys
import argparse
from datetime import datetime
import math

import pandas as pd
from numpy.random import default_rng
import simpy
from pathlib import Path


class InterviewSessions(object):
    def __init__(self, env, num_reg_staff, num_screening_staff, num_interviewers, num_schedulers,
                 mean_interarrival_time, pct_second_round,
                 reg_time_mean,document_check_time_mean, document_check_time_sd,
                 interview_time_mean, interview_time_sd,
                 sched_time_mean, sched_time_sd,
                 post_int_time, add_post_int_time_mean, rg
                 ):
        """
        Primary class that encapsulates InterviewSessions resources.

        The detailed interview flow logic is in interview_scheduled() function.

        Parameters
        ----------
        env
        num_reg_staff
        num_screening_staff
        num_interviewers
        num_schedulers
        mean_interarrival_time
        pct_second_round
        reg_time_mean
        document_check_time_mean
        document_check_time_sd
        interview_time_mean
        interview_time_sd
        sched_time_mean
        sched_time_sd
        post_int_time
        add_post_int_time_mean
        rg
        """

        # Simulation environment and random number generator
        self.env = env
        self.rg = rg

        # Create list to hold timestamps dictionaries (one per candidate)
        self.timestamps_list = []
        # Create lists to hold occupancy tuples (time, occ)
        self.postint_occupancy_list = [(0.0, 0.0)]
        self.int_occupancy_list = [(0.0, 0.0)]

        # Create SimPy resources
        self.reg_staff = simpy.Resource(env, num_reg_staff)
        self.screening_staff = simpy.Resource(env, num_screening_staff)
        self.interviewer = simpy.Resource(env, num_interviewers)
        self.scheduler = simpy.Resource(env, num_schedulers)

        # Initialize the candidate flow related attributes
        self.mean_interarrival_time = mean_interarrival_time
        self.pct_second_round = pct_second_round

        self.reg_time_mean = reg_time_mean
        self.document_check_time_mean = document_check_time_mean
        self.document_check_time_sd = document_check_time_sd
        self.interview_time_mean = interview_time_mean
        self.interview_time_sd = interview_time_sd
        self.sched_time_mean = sched_time_mean
        self.sched_time_sd = sched_time_sd
        self.post_int_time = post_int_time
        self.add_post_int_time_mean = add_post_int_time_mean


    # Create process methods

    def registration(self):
        yield self.env.timeout(self.rg.exponential(self.reg_time_mean))

    def document_check(self):
        yield self.env.timeout(self.rg.normal(self.document_check_time_mean, self.document_check_time_sd))

    def interview(self):
        yield self.env.timeout(self.rg.normal(self.interview_time_mean, self.interview_time_sd))

    def schedule_next_round(self):
        yield self.env.timeout(self.rg.normal(self.sched_time_mean, self.sched_time_sd))

    # We assume all candidates wait at least obs_time minutes post-interview
    # Some might have to wait longer. This is the time beyond post_int_time minutes
    # that candidates wait.
    def wait_gt_post_int_time(self):
        yield self.env.timeout(self.rg.exponential(self.add_post_int_time_mean))


def interview_sheduled(env, candidate, interviewprocess, quiet):
    """Defines the sequence of steps traversed by candidates.

       Also capture a bunch of timestamps to make it easy to compute various system
       performance measures such as candidate waiting times, queue sizes and resource utilization.
    """
    # Candidate arrives to the facility
    arrival_ts = env.now


    # Request reg staff to get registered
    with interviewprocess.reg_staff.request() as request:
        yield request
        got_reg_ts = env.now
        yield env.process(interviewprocess.registration())
        release_reg_ts = env.now

    with interviewprocess.screening_staff.request() as request:
        yield request
        # Now that we have a screening staff, check documents. Note time.
        got_screening_staff_ts = env.now
        yield env.process(interviewprocess.document_check())
        release_screening_staff_ts = env.now

    # Request interviewer for interview
    with interviewprocess.interviewer.request() as request:
        if not quiet:
            print(f"Candidate {candidate} requests interviewer at time {env.now}")
        yield request
        got_interviewer_ts = env.now
        q_time = got_interviewer_ts - release_screening_staff_ts
        if not quiet:
            print(f"Candidate {candidate} gets interviewer at time {env.now} (waited {q_time:.1f} minutes)")
        # Update int occupancy - increment by 1
        prev_occ = interviewprocess.int_occupancy_list[-1][1]
        new_occ = (env.now, prev_occ + 1)
        interviewprocess.int_occupancy_list.append(new_occ)
        yield env.process(interviewprocess.interview())
        release_interviewer_ts = env.now
        if not quiet:
            print(f"Candiadate {candidate} releases interviewer at time {env.now}")

        interviewprocess.int_occupancy_list.append((env.now, interviewprocess.int_occupancy_list[-1][1] - 1))

        # Update postint occupancy - increment by 1
        interviewprocess.postint_occupancy_list.append((env.now, interviewprocess.postint_occupancy_list[-1][1] + 1))

    # Request scheduler to schedule second round of interview if needed
    if interviewprocess.rg.random() < interviewprocess.pct_second_round:
        with interviewprocess.scheduler.request() as request:
            yield request
            got_scheduler_ts = env.now
            yield env.process(interviewprocess.schedule_next_round())
            release_scheduler_ts = env.now
    else:
        got_scheduler_ts = pd.NA
        release_scheduler_ts = pd.NA

    # Wait at least post_int_time minutes from time interview is completed
    post_interv_time = env.now - release_interviewer_ts
    if post_interv_time < interviewprocess.post_int_time:
        # Wait until 10 total minutes post interview
        yield env.timeout(interviewprocess.post_int_time - post_interv_time)
        # Wait random amount beyond post_int_time
        yield env.process(interviewprocess.wait_gt_post_int_time())

        # Update postint occupancy - decrement by 1
        interviewprocess.postint_occupancy_list.append((env.now, interviewprocess.postint_occupancy_list[-1][1] - 1))

    exit_system_ts = env.now
    if not quiet:
        print(f"Candidate {candidate} exited system at time {env.now}")

    # Create dictionary of timestamps
    timestamps = {'candidate_id': candidate,
                  'arrival_ts': arrival_ts,
                  'got_reg_ts': got_reg_ts,
                  'release_reg_ts': release_reg_ts,
                  'got_screening_staff_ts': got_screening_staff_ts,
                  'release_screening_staff_ts': release_screening_staff_ts,
                  'got_interviewer_ts': got_interviewer_ts,
                  'release_interviewer_ts': release_interviewer_ts,
                  'got_scheduler_ts': got_scheduler_ts,
                  'release_scheduler_ts': release_scheduler_ts,
                  'exit_system_ts': exit_system_ts}

    interviewprocess.timestamps_list.append(timestamps)


def run_interviewprocess(env, interviewprocess, stoptime=simpy.core.Infinity, max_arrivals=simpy.core.Infinity, quiet=False):
    """
    Run the interview for a specified amount of time or after generating a maximum number of candidates.

    Parameters
    ----------
    env : SimPy environment
    interviewprocess : ``InterviewSessions`` object
    stoptime : float
    max_arrivals : int
    quiet : bool

    Yields
    -------
    Simpy environment timeout
    """

    # Create a counter to keep track of number of candiadtes generated and to serve as unique candidate id
    candidate = 0

    # Loop for generating candidates
    while env.now < stoptime and candidate < max_arrivals:
        # Generate next interarrival time
        iat = interviewprocess.rg.exponential(interviewprocess.mean_interarrival_time)

        # This process will now yield to a 'timeout' event. This process will resume after iat time units.
        yield env.timeout(iat)

        # New candidate generated = update counter of candidates
        candidate += 1

        if not quiet:
            print(f"Candidate {candidate} created at time {env.now}")

        # Register a interview_scheduled process for the new candidate
        env.process(interview_sheduled(env, candidate, interviewprocess, quiet))

    print(f"{candidate} candidates processed.")


def compute_durations(timestamp_df):
    """Compute time durations of interest from timestamps dataframe and append new cols to dataframe"""

    timestamp_df['wait_for_reg'] = timestamp_df.loc[:, 'got_reg_ts'] - timestamp_df.loc[:, 'arrival_ts']

    timestamp_df['wait_for_screening_staff'] = timestamp_df.loc[:, 'got_screening_staff_ts'] - timestamp_df.loc[:,
                                                                                               'release_reg_ts']

    timestamp_df['wait_for_interviewer'] = timestamp_df.loc[:, 'got_interviewer_ts'] - timestamp_df.loc[:,
                                                                                       'release_screening_staff_ts']

    timestamp_df['interview_time'] = timestamp_df.loc[:, 'release_interviewer_ts'] - timestamp_df.loc[:,
                                                                                     'got_interviewer_ts']
    timestamp_df['wait_for_scheduler'] = timestamp_df.loc[:, 'got_scheduler_ts'] - timestamp_df.loc[:,
                                                                                   'release_interviewer_ts']
    timestamp_df['post_int_time'] = timestamp_df.loc[:, 'exit_system_ts'] - timestamp_df.loc[:,
                                                                             'release_interviewer_ts']
    timestamp_df['time_in_system'] = timestamp_df.loc[:, 'exit_system_ts'] - timestamp_df.loc[:, 'arrival_ts']

    return timestamp_df


def simulate(arg_dict, rep_num):
    """

    Parameters
    ----------
    arg_dict : dict whose keys are the input args
    rep_num : int, simulation replication number

    Returns
    -------
    Nothing returned but numerous output files written to ``args_dict[output_path]``

    """

    # Create a random number generator for this replication
    seed = arg_dict['seed'] + rep_num - 1
    rg = default_rng(seed=seed)

    # Resource capacity levels
    num_reg_staff = arg_dict['num_reg_staff']
    num_screening_staff = arg_dict['num_screening_staff']
    num_interviewers = arg_dict['num_interviewers']
    num_schedulers = arg_dict['num_schedulers']

    # Initialize the candidate flow related attributes
    candidate_arrival_rate = arg_dict['candidate_arrival_rate']
    mean_interarrival_time = 1.0 / (candidate_arrival_rate / 60.0)

    pct_second_round = arg_dict['pct_second_round']
    reg_time_mean = arg_dict['reg_time_mean']
    document_check_time_mean = arg_dict['document_check_time_mean']
    document_check_time_sd = arg_dict['document_check_time_sd']
    interview_time_mean = arg_dict['interview_time_mean']
    interview_time_sd = arg_dict['interview_time_sd']
    sched_time_mean = arg_dict['sched_time_mean']
    sched_time_sd = arg_dict['sched_time_sd']
    post_int_time = arg_dict['post_int_time']
    add_post_int_time_mean = arg_dict['add_post_int_time_mean']

    # Other parameters
    stoptime = arg_dict['stoptime']  # No more arrivals after this time
    quiet = arg_dict['quiet']
    scenario = arg_dict['scenario']

    # Run the simulation
    env = simpy.Environment()

    # Create a interviewprocess to simulate
    interviewprocess = InterviewSessions(env, num_reg_staff,num_screening_staff, num_interviewers, num_schedulers,
                           mean_interarrival_time, pct_second_round,
                           reg_time_mean,
                               document_check_time_mean, document_check_time_sd,
                           interview_time_mean, interview_time_sd,
                           sched_time_mean, sched_time_sd,
                           post_int_time, add_post_int_time_mean, rg
                           )

    # Initialize and register the run_interviewprocess generator function
    env.process(
        run_interviewprocess(env, interviewprocess, stoptime=stoptime, quiet=quiet))

    # Launch the simulation
    env.run()

    # Create output files and basic summary stats
    if len(arg_dict['output_path']) > 0:
        output_dir = Path.cwd() / arg_dict['output_path']
    else:
        output_dir = Path.cwd()

    # Create paths for the output logs
    interviewprocess_candidate_log_path = output_dir / f'interviewprocess_candidate_{scenario}_{rep_num}.csv'
    int_occupancy_df_path = output_dir / f'int_occupancy_{scenario}_{rep_num}.csv'
    postint_occupancy_df_path = output_dir / f'postint_occupancy_{scenario}_{rep_num}.csv'

    # Create candidate log dataframe and add scenario and rep number cols
    interviewprocess_candidate_log_df = pd.DataFrame(interviewprocess.timestamps_list)
    interviewprocess_candidate_log_df['scenario'] = scenario
    interviewprocess_candidate_log_df['rep_num'] = rep_num

    # Reorder cols to get scenario and rep_num first
    num_cols = len(interviewprocess_candidate_log_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols-2)])
    interviewprocess_candidate_log_df = interviewprocess_candidate_log_df.iloc[:, new_col_order]

    # Compute durations of interest for candidate log
    interviewprocess_candidate_log_df = compute_durations(interviewprocess_candidate_log_df)

    # Create occupancy log dataframes and add scenario and rep number cols
    int_occupancy_df = pd.DataFrame(interviewprocess.int_occupancy_list, columns=['ts', 'occ'])
    int_occupancy_df['scenario'] = scenario
    int_occupancy_df['rep_num'] = rep_num
    num_cols = len(int_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    int_occupancy_df = int_occupancy_df.iloc[:, new_col_order]

    postint_occupancy_df = pd.DataFrame(interviewprocess.postint_occupancy_list, columns=['ts', 'occ'])
    postint_occupancy_df['scenario'] = scenario
    postint_occupancy_df['rep_num'] = rep_num
    num_cols = len(postint_occupancy_df.columns)
    new_col_order = [-2, -1]
    new_col_order.extend([_ for _ in range(0, num_cols - 2)])
    postint_occupancy_df = int_occupancy_df.iloc[:, new_col_order]

    # Export logs to csv
    interviewprocess_candidate_log_df.to_csv(interviewprocess_candidate_log_path, index=False)
    int_occupancy_df.to_csv(int_occupancy_df_path, index=False)
    postint_occupancy_df.to_csv(postint_occupancy_df_path, index=False)

    # Note simulation end time
    end_time = env.now
    print(f"Simulation replication {rep_num} ended at time {end_time}")


def process_sim_output(csvs_path, scenario):
    """

    Parameters
    ----------
    csvs_path : Path object for location of simulation output candidate log csv files
    scenario : str

    Returns
    -------
    Dict of dicts

    Keys are:

    'candidate_log_rep_stats' --> Contains dataframes from describe on group by rep num. Keys are perf measures.
    'candidate_log_ci' -->        Contains dictionaries with overall stats and CIs. Keys are perf measures.
    """

    dest_path = csvs_path / f"consolidated_interviewprocess_canidate_log_{scenario}.csv"

    sort_keys = ['scenario', 'rep_num']

    # Create empty dict to hold the DataFrames created as we read each csv file
    dfs = {}

    # Loop over all the csv files
    for csv_f in csvs_path.glob('interviewprocess_candidate*.csv'):
        # Split the filename off from csv extension. We'll use the filename

        fstem = csv_f.stem

        # Read the next csv file into a pandas DataFrame and add it to
        # the dfs dict.
        df = pd.read_csv(csv_f)
        dfs[fstem] = df

    # Use pandas concat method to combine the file specific DataFrames into
    # one big DataFrame.
    candidate_log_df = pd.concat(dfs)

    candidate_log_df.sort_values(sort_keys, inplace=True)

    # Export the final DataFrame to a csv file. Suppress the pandas index.
    candidate_log_df.to_csv(dest_path, index=False)

    # Compute summary statistics for several performance measures
    candidate_log_stats = summarize_candidate_log(candidate_log_df, scenario)

    # Now delete the individual replication files
    for csv_f in csvs_path.glob('interviewprocess_candidate*.csv'):
        csv_f.unlink()

    return candidate_log_stats


def summarize_candidate_log(candidate_log_df, scenario):
    """

    Parameters
    ----------
    candidate_log_df : DataFrame created by process_sim_output
    scenario : str

    Returns
    -------
    Dict of dictionaries - See comments below
    """

    # Create empty dictionaries to hold computed results
    candidate_log_rep_stats = {}  # Will store dataframes from describe on group by rep num. Keys are perf measures.
    candidate_log_ci = {}         # Will store dictionaries with overall stats and CIs. Keys are perf measures.
    candidate_log_stats = {}      # Container dict returned by this function containing the two previous dicts.

    # Create list of performance measures for looping over
    performance_measures = ['wait_for_reg', 'wait_for_screening_staff', 'wait_for_interviewer',
                           'wait_for_scheduler', 'time_in_system']

    for pm in performance_measures:
        # Compute descriptive stats for each replication and store dataframe in dict
        candidate_log_rep_stats[pm] = candidate_log_df.groupby(['rep_num'])[pm].describe()
        # Compute across replication stats
        n_samples = candidate_log_rep_stats[pm]['mean'].count()
        mean_mean = candidate_log_rep_stats[pm]['mean'].mean()
        sd_mean = candidate_log_rep_stats[pm]['mean'].std()
        ci_95_lower = mean_mean - 1.96 * sd_mean / math.sqrt(n_samples)
        ci_95_upper = mean_mean + 1.96 * sd_mean / math.sqrt(n_samples)
        # Store cross replication stats as dict in dict
        candidate_log_ci[pm] = {'n_samples': n_samples, 'mean_mean': mean_mean, 'sd_mean': sd_mean,
                              'ci_95_lower': ci_95_lower, 'ci_95_upper': ci_95_upper}

    candidate_log_stats['scenario'] = scenario
    candidate_log_stats['candidate_log_rep_stats'] = candidate_log_rep_stats
    # Convert the final summary stats dict to a DataFrame
    candidate_log_stats['candidate_log_ci'] = pd.DataFrame(candidate_log_ci)

    return candidate_log_stats


def process_command_line():
    """
    Parse command line arguments

    `argv` is a list of arguments, or `None` for ``sys.argv[1:]``.
    Return a Namespace representing the argument list.
    """

    # Create the parser
    parser = argparse.ArgumentParser(prog='interview_simulation_model',
                                     description='Run interview simulation')

    # Add arguments
    parser.add_argument(
        "--config", type=str, default=None,
        help="Configuration file containing input parameter arguments and values"
    )

    parser.add_argument("--candidate_arrival_rate", default=150, help="candidates per hour",
                        type=float)

    parser.add_argument("--num_reg_staff", default=2, help="number of registration staff",
                        type=int)

    parser.add_argument("--num_screening_staff", default=2, help="number of screening_staff",
                        type=int)

    parser.add_argument("--num_interviewers", default=15, help="number of interviewers",
                        type=int)

    parser.add_argument("--num_schedulers", default=2, help="number of schedulers",
                        type=int)

    parser.add_argument("--pct_second_round", default=0.5,
                        help="percent of candidates needing next round of interview (default = 0.5)",
                        type=float)

    parser.add_argument("--reg_time_mean", default=1.0,
                        help="Mean time (mins) for registration (default = 1.0)",
                        type=float)

    parser.add_argument("--document_check_time_mean", default=0.25,
                        help="Mean time (mins) for documents check (default = 0.25)",
                        type=float)

    parser.add_argument("--document_check_time_sd", default=0.05,
                        help="Standard deviation time (mins) for documents check (default = 0.05)",
                        type=float)

    parser.add_argument("--interview_time_mean", default=4.0,
                        help="Mean time (mins) for interview (default = 4.0)",
                        type=float)

    parser.add_argument("--interview_time_sd", default=0.5,
                        help="Standard deviation time (mins) for interview (default = 0.5)",
                        type=float)

    parser.add_argument("--sched_time_mean", default=1.0,
                        help="Mean time (mins) for scheduling next round of interview (default = 1.0)",
                        type=float)

    parser.add_argument("--sched_time_sd", default=0.1,
                        help="Standard deviation time (mins) for scheduling next round of interview (default = 0.1)",
                        type=float)

    parser.add_argument("--post_int_time", default=15.0,
                        help="Time (minutes) candidate waits post interview in waiting area for results (default = 15)",
                        type=float)

    parser.add_argument("--add_post_int_time_mean", default=1.0,
                        help="Time (minutes) candidate waits post waiting time in waiting area (default = 1.0)",
                        type=float)

    parser.add_argument(
        "--scenario", type=str, default=datetime.now().strftime("%Y.%m.%d.%H.%M."),
        help="Appended to output filenames."
    )

    parser.add_argument("--stoptime", default=600, help="time that simulation stops (default = 600)",
                        type=float)

    parser.add_argument("--num_reps", default=1, help="number of simulation replications (default = 1)",
                        type=int)

    parser.add_argument("--seed", default=3, help="random number generator seed (default = 3)",
                        type=int)

    parser.add_argument(
        "--output_path", type=str, default="", help="location for output file writing")

    parser.add_argument("--quiet", action='store_true',
                        help="If True, suppresses output messages (default=False")

    # do the parsing
    args = parser.parse_args()

    if args.config is not None:
        # Read inputs from config file
        with open(args.config, "r") as fin:
            args = parser.parse_args(fin.read().split())

    return args


def main():

    args = process_command_line()
    print(args)

    num_reps = args.num_reps
    scenario = args.scenario

    if len(args.output_path) > 0:
        output_dir = Path.cwd() / args.output_path
    else:
        output_dir = Path.cwd()

    # Main simulation replication loop
    for i in range(1, num_reps + 1):
        simulate(vars(args), i)

    # Consolidate the candidate logs and compute summary stats
    candidate_log_stats = process_sim_output(output_dir, scenario)
    print(f"\nScenario: {scenario}")
    pd.set_option("display.precision", 3)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 120)
    print(candidate_log_stats['candidate_log_rep_stats'])
    print(candidate_log_stats['candidate_log_ci'])


if __name__ == '__main__':
    main()


