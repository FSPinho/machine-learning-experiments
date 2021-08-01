import copy
import datetime
import json

DEF_COLOR_INFO = '\033[0;34m'
DEF_COLOR_WARN = '\033[0;33m'
DEF_COLOR_ERROR = '\033[0;31m'
DEF_COLOR_RESET = '\033[0m'

DEF_INDENTATION_CHAR = '    '
DEF_LOG_LEVEL_LABELS = {
    1: '%sINFO %s' % (DEF_COLOR_INFO, DEF_COLOR_RESET),
    2: '%sWARN %s' % (DEF_COLOR_WARN, DEF_COLOR_RESET),
    3: '%sERROR%s' % (DEF_COLOR_ERROR, DEF_COLOR_RESET),
}


class LogLevel:
    INFO = 1
    WARNING = 2
    ERROR = 3


# noinspection PyBroadException
class Log:
    @staticmethod
    def readable_size(num, suffix="B"):
        for unit in ["", "K", "M", "G", "T", "P", "E", "Z"]:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, "Y", suffix)

    @staticmethod
    def _is_serializable(obj):
        try:
            json.dumps(obj)
            return True
        except TypeError:
            return False

    @staticmethod
    def _clear_obj(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, datetime.date):
                    obj[k] = v.isoformat()
                else:
                    obj[k] = Log._clear_obj(obj[k])

        elif isinstance(obj, (list, tuple)):
            return [Log._clear_obj(_item) for _item in obj]

        elif not Log._is_serializable(obj):
            try:
                text = str(obj)
                return text
            except Exception:
                return ' - NOT SERIALIZABLE - '

        return obj

    @staticmethod
    def _clear_text(text):
        if isinstance(text, str):
            return text

        try:
            text = Log._clear_obj(copy.deepcopy(text))
        except Exception:
            text = Log._clear_obj(text)

        return json.dumps(text, indent=4)

    @staticmethod
    def log(*args, **kwargs):

        """
        Method intended to show a log message on terminal only
        for development purposes. The message will be showed
        only if DEBUG is enabled on settings.
        :param args: The log message parts.
        :param kwargs:
            - log_level: The message level type. See LogLevel class for more.
            - indent: Amount if indentation at left of the message.
        """

        log_level = kwargs.get('log_level', LogLevel.INFO)
        indent = kwargs.get('indent', 1)
        _log_level_label = DEF_LOG_LEVEL_LABELS[log_level]
        _indentation = ''.join(map(lambda x: DEF_INDENTATION_CHAR, range(max(0, indent - 1))))
        print('%s %s %s' % (_log_level_label, _indentation, ' '.join(map(lambda x: str(Log._clear_text(x)), args))))

    @staticmethod
    def i(*args, **kwargs):
        Log.log(*args, log_level=LogLevel.INFO, **kwargs)

    @staticmethod
    def w(*args, **kwargs):
        Log.log(*args, log_level=LogLevel.WARNING, **kwargs)

    @staticmethod
    def e(*args, **kwargs):
        Log.log(*args, log_level=LogLevel.ERROR, **kwargs)
