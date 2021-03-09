from qa         import qa_main
from trade      import trade_main
from stats      import stats_main
from backtest   import backtest_main
from util       import open_logfile, log, get_current_day_and_time, is_holiday
from symbols    import LOGPATH
from time       import time, sleep


def calc_time_passed(start_time):
    # needs to go into util.py as helper function
    seconds = int(time() - start_time)
    minutes = int(seconds / 60)
    seconds = seconds - minutes * 60
    hours = int(minutes / 60)
    minutes = minutes - hours * 60
    return seconds, minutes, hours

def job_main():
    start_time = time()
    log('Starting main Stockie job.')


    qa_start_time = time()
    log('Starting qa_main().')
    qa_main()
    log('qa_main() is done.')
    seconds, minutes, hours = calc_time_passed(qa_start_time)
    log(f'qa_main() took (hh:mm:ss) : {hours}:{minutes}:{seconds}')


    trade_start_time = time()
    log('Starting trade_main().')
    trade_main()
    log('trade_main() is done.')
    seconds, minutes, hours = calc_time_passed(trade_start_time)
    log(f'trade_main() took (hh:mm:ss) : {hours}:{minutes}:{seconds}')


    stats_start_time = time()
    log('Starting stats_main().')
    stats_main()
    log('stats_main() is done.')
    seconds, minutes, hours = calc_time_passed(stats_start_time)
    log(f'trade_main() took (hh:mm:ss) : {hours}:{minutes}:{seconds}')


    backtest_start_time = time()
    log('Starting backtest_main().')
    backtest_main()
    log('backtest_main() is done.')
    seconds, minutes, hours = calc_time_passed(backtest_start_time)
    log(f'trade_main() took (hh:mm:ss) : {hours}:{minutes}:{seconds}')


    log('')
    log('Done.')

    # needs to go into util.py as helper function
    sleep(60)
    seconds, minutes, hours = calc_time_passed(start_time)
    log(f'Run time (hh:mm:ss) : {hours}:{minutes}:{seconds}')

if __name__ == "__main__":

    log_fnm = "job"+ get_current_day_and_time() + ".log"
    open_logfile(LOGPATH, log_fnm)

    if is_holiday() == True:
        log('Today is not a trading day!', True)
        log('', True)
        log('Done', True)
    else:
        job_main()