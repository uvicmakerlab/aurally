import datetime
import logging
import functools

logger = logging.getLogger('debug_trace')

def configure(filename='debug.log', format='%(levelname)s: %(message)s', level=logging.WARNING):
    logger.setLevel(level)

    formatter = logging.Formatter(format)

    fh = logging.FileHandler(filename)
    fh.setLevel(level)
    fh.setFormatter(formatter)

    logger.addHandler(fh)
    logger.info('Starting log at {0}...'.format(datetime.datetime.now()))

def trace(level):
    def _decorator(f):
        @functools.wraps(f)
        def _decorated(*args, **kwds):
            logger.log(level, 'Enter {0}({1}, {2})'.format(f.func_name, args, kwds))
            ans = None
            try:
                ans = f(*args, **kwds)
            except:
                logger.log(level, 'Exit {0}, exception thrown.'.format(f.func_name))
                raise
            logger.log(level, 'Exit {0}, returned {1}'.format(f.func_name, ans))
            return ans
        return _decorated
    return _decorator

