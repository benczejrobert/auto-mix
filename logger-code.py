import logging
import sys

# TODO check answer with 48 upvotes:
# https://stackoverflow.com/questions/14058453/making-python-loggers-output-all-messages-to-stdout-in-addition-to-log-file

# 1) want to output to console too - DONE
# 2) want to read from log file while it's being written - almost realtime. maybe make bash script that gets last line from file or sth
# https://stackoverflow.com/questions/3290292/read-from-a-log-file-as-its-being-written-using-python
# 3) want to add exceptions to log file as well - DONE (along with 4)

# 4) (optional) add ALL prints' output to the log - DONE:
# https://stackoverflow.com/questions/11124093/redirect-python-print-output-to-logger
# wouldn't necessarily need coz it'd be too verbose at times MAYBE - for instance, with the prints in the scaler but we'll see

# 5) make folders of the files for each run AND create a new file after each N rows written.

# TODO check how to read from log file while it's being written
# file_handler = logging.FileHandler(filename='temp.log') # TODO change path to sth else AND add crt run datetime
# stdout_handler = logging.StreamHandler(stream=sys.stdout)
# handlers = [file_handler, stdout_handler]
#
# logging.basicConfig(
#     level=logging.DEBUG,
#     format='[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s',
#     handlers=handlers
# )
#
# logger = logging.getLogger('LOGGER_NAME')
#
# # todo checkif this logger cam be used for multiple files
# # check howto capture all console output for minimal code changes (i.e. any print msg)
# print("lol")
# logger.info('Starting something...')


###########3 other code

#
# import logging
# import auxiliary_module
#
# # create logger with 'spam_application'
# log = logging.getLogger('spam_application')
# log.setLevel(logging.DEBUG)
#
# # create formatter and add it to the handlers
# formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#
# # create file handler which logs even debug messages
# fh = logging.FileHandler('spam.log')
# fh.setLevel(logging.DEBUG)
# fh.setFormatter(formatter)
# log.addHandler(fh)
#
# # create console handler with a higher log level
# ch = logging.StreamHandler()
# ch.setLevel(logging.ERROR)
# ch.setFormatter(formatter)
# log.addHandler(ch)
#
# log.info('creating an instance of auxiliary_module.Auxiliary')
# a = auxiliary_module.Auxiliary()
# log.info('created an instance of auxiliary_module.Auxiliary')
#
# log.info('calling auxiliary_module.Auxiliary.do_something')
# a.do_something()
# log.info('finished auxiliary_module.Auxiliary.do_something')
#
# log.info('calling auxiliary_module.some_function()')
# auxiliary_module.some_function()
# log.info('done with auxiliary_module.some_function()')
#
# # remember to close the handlers
# for handler in log.handlers:
#     handler.close()
#     log.removeFilter(handler)

# Please see: https://docs.python.org/2/howto/logging-cookbook.html
