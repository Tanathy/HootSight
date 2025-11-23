import datetime
import os
import inspect
import sys
import time
import re
from contextlib import contextmanager

from threading import Lock
from collections import OrderedDict

_progress_lock = Lock()

def clear() -> None:
    sys.stdout.flush()
    sys.stderr.flush()
    if os.name == 'nt':
        os.system('cls')
    else:
        os.system('clear')

class Colors:
    RESET = '\033[0m'
    INFO = '\033[37m'      # White (normal)
    WARNING = '\033[33m'   # Yellow
    ERROR = '\033[31m'     # Red
    SUCCESS = '\033[32m'   # Green
    
    PROGRESS_COMPLETE = '\033[42m'  # Green background
    PROGRESS_INCOMPLETE = '\033[47m'  # White background
    
    CYAN = '\033[36m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'

class ProgressChars:
    FULL = '█'
    SEVEN_EIGHTHS = '▉' 
    THREE_QUARTERS = '▊'
    FIVE_EIGHTHS = '▋'
    HALF = '▌'
    THREE_EIGHTHS = '▍'
    QUARTER = '▎'
    EIGHTH = '▏'
    EMPTY = ' '
    
    ASCII_FULL = '#'
    ASCII_EMPTY = '-'

if os.name == 'nt':
    try:
        import ctypes
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE = -11
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(handle, mode)
        _ansi_supported = True
    except Exception:
        _ansi_supported = False
else:
    _ansi_supported = True

def _strip_ansi(text):
    return re.sub(r'\033\[[0-9;]*[mK]', '', text)

def _write_ansi(text):
    if _ansi_supported:
        sys.stdout.write(text)
    else:
        sys.stdout.write(_strip_ansi(text))

def get_caller_name() -> str:
    frame = inspect.currentframe()
    try:
        current_frame = frame
        for _ in range(3):
            if current_frame and current_frame.f_back:
                current_frame = current_frame.f_back
            else:
                return 'unknown'
        
        if current_frame:
            filename = current_frame.f_code.co_filename
            script_name = os.path.basename(filename).replace('.py', '')
            
            if script_name in ['log', 'coordinator_settings', '__main__']:
                if current_frame.f_back:
                    filename = current_frame.f_back.f_code.co_filename
                    script_name = os.path.basename(filename).replace('.py', '')
            
            return script_name
        
        return 'unknown'
    finally:
        del frame

def format_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m{secs:02d}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h{minutes:02d}m"

def format_rate(rate: float, unit: str = 'it') -> str:
    if rate <= 0:
        return f"0.0{unit}/s"
    elif rate < 1:
        return f"{1/rate:.1f}s/{unit}"
    elif rate < 1000:
        return f"{rate:.1f}{unit}/s"
    elif rate < 1000000:
        return f"{rate/1000:.1f}k{unit}/s"
    else:
        return f"{rate/1000000:.1f}M{unit}/s"

def log(message: str, level: str = 'INFO') -> None:
    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    script_name = get_caller_name()
    level = level.upper()
    
    if level == 'WARNING':
        color = Colors.WARNING
    elif level == 'ERROR':
        color = Colors.ERROR
    elif level == 'SUCCESS':
        color = Colors.SUCCESS
    else:
        color = Colors.INFO
    
    formatted_message = f"{timestamp}\t{level}\t{script_name}\t{message}"
    
    with _progress_lock:
        print(f"{color}{formatted_message}{Colors.RESET}")

def info(message: str) -> None:
    log(message, 'INFO')

def warning(message: str) -> None:
    log(message, 'WARNING')

def error(message: str) -> None:
    log(message, 'ERROR')

def success(message: str) -> None:
    log(message, 'SUCCESS')

class ProgressBar:
    def __init__(self, total=None, desc='', unit='it', unit_scale=False, 
                 ncols=None, colour='cyan', ascii_fallback=False, 
                 leave=True, miniters=1, mininterval=0.1):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.unit_scale = unit_scale
        self.ncols = ncols or self._get_terminal_width()
        self.colour = getattr(Colors, colour.upper(), Colors.CYAN)
        self.ascii_fallback = ascii_fallback
        self.leave = leave
        self.miniters = miniters
        self.mininterval = mininterval
        
        self.n = 0
        self.last_print_n = 0
        self.start_time = time.time()
        self.last_print_time = self.start_time
        
        self.closed = False
        self.displayed = False
        
        self._rate_history = []
        self._rate_window = 10  # Number of samples for rate calculation
        
    def _get_terminal_width(self):
        try:
            import shutil
            return shutil.get_terminal_size().columns
        except:
            return 80
    
    def _supports_unicode(self):
        if self.ascii_fallback:
            return False
        try:
            ProgressChars.FULL.encode(sys.stdout.encoding or 'utf-8')
            return True
        except:
            return False
    
    def _format_bar(self, frac, bar_length):
        if self._supports_unicode():
            chars = [ProgressChars.EMPTY, ProgressChars.EIGHTH, ProgressChars.QUARTER,
                    ProgressChars.THREE_EIGHTHS, ProgressChars.HALF, ProgressChars.FIVE_EIGHTHS,
                    ProgressChars.THREE_QUARTERS, ProgressChars.SEVEN_EIGHTHS, ProgressChars.FULL]
            
            nsyms = len(chars) - 1
            full_length = int(frac * bar_length * nsyms)
            full_bars = full_length // nsyms
            remainder = full_length % nsyms
            
            bar = chars[-1] * full_bars
            if full_bars < bar_length:
                bar += chars[remainder]
                bar += chars[0] * (bar_length - full_bars - 1)
        else:
            full_char = ProgressChars.ASCII_FULL
            empty_char = ProgressChars.ASCII_EMPTY
            filled_length = int(bar_length * frac)
            bar = full_char * filled_length + empty_char * (bar_length - filled_length)
        
        return f"{self.colour}{bar}{Colors.RESET}"
    
    def _format_number(self, n):
        if not self.unit_scale:
            return str(int(n)) if n == int(n) else f"{n:.1f}"
        
        if n < 1000:
            return str(int(n)) if n == int(n) else f"{n:.1f}"
        elif n < 1000000:
            return f"{n/1000:.1f}k"
        elif n < 1000000000:
            return f"{n/1000000:.1f}M"
        else:
            return f"{n/1000000000:.1f}G"
    
    def _calculate_rate(self):
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if elapsed > 0:
            rate = self.n / elapsed
            
            self._rate_history.append(rate)
            if len(self._rate_history) > self._rate_window:
                self._rate_history.pop(0)
            
            return sum(self._rate_history) / len(self._rate_history)
        
        return 0
    
    def _format_meter(self):
        elapsed = time.time() - self.start_time
        
        if self.total:
            frac = min(self.n / self.total, 1.0)
            percentage = frac * 100
        else:
            frac = 0
            percentage = 0
        
        desc_str = f"{self.desc}: " if self.desc else ""
        
        if self.total:
            n_fmt = self._format_number(self.n)
            total_fmt = self._format_number(self.total)
            percentage_str = f"{percentage:3.0f}%"
            
            prefix_len = len(f"{desc_str}{percentage_str}|")
            suffix_len = len(f"| {n_fmt}/{total_fmt} [{format_time(elapsed)}<")
            
            rate = self._calculate_rate()
            if rate > 0 and self.total:
                remaining_time = (self.total - self.n) / rate
                time_left = format_time(remaining_time)
            else:
                time_left = "?"
            
            rate_str = format_rate(rate, self.unit)
            suffix_len += len(f"{time_left}, {rate_str}]")
            
            bar_length = max(10, self.ncols - prefix_len - suffix_len)
            bar = self._format_bar(frac, bar_length)
            
            return (f"{desc_str}{percentage_str}|{bar}| "
                   f"{n_fmt}/{total_fmt} [{format_time(elapsed)}<{time_left}, {rate_str}]")
        else:
            n_fmt = self._format_number(self.n)
            rate = self._calculate_rate()
            rate_str = format_rate(rate, self.unit)
            
            return f"{desc_str}{n_fmt}{self.unit} [{format_time(elapsed)}, {rate_str}]"
    
    def update(self, n=1):
        if self.closed:
            return
        
        self.n += n
        current_time = time.time()
        
        if (self.n - self.last_print_n >= self.miniters and 
            current_time - self.last_print_time >= self.mininterval):
            
            self.refresh()
            self.last_print_n = self.n
            self.last_print_time = current_time
    
    def refresh(self):
        if self.closed:
            return
            
        with _progress_lock:
            meter = self._format_meter()
            if self.displayed:
                print(f"\r{meter}", end='', flush=True)
            else:
                print(meter, end='', flush=True)
                self.displayed = True
    
    def set_description(self, desc):
        self.desc = desc
        self.refresh()
    
    def set_postfix(self, **kwargs):
        if kwargs:
            postfix_str = ', '.join(f"{k}={v}" for k, v in kwargs.items())
            if not hasattr(self, '_original_desc'):
                self._original_desc = self.desc
            self.desc = f"{self._original_desc} [{postfix_str}]"
            self.refresh()
    
    def close(self):
        if self.closed:
            return
        
        self.closed = True
        
        with _progress_lock:
            if self.displayed:
                if self.leave:
                    print()
                else:
                    print(f"\r{' ' * self.ncols}\r", end='', flush=True)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def progress_bar(iterable=None, total=None, **kwargs):
    if iterable is not None:
        if total is None:
            try:
                total = len(iterable)
            except TypeError:
                total = None
        
        pbar = ProgressBar(total=total, **kwargs)
        
        class ProgressWrapper:
            def __init__(self, iterable, pbar):
                self.iterable = iterable
                self.pbar = pbar
            
            def __iter__(self):
                try:
                    for item in self.iterable:
                        yield item
                        self.pbar.update(1)
                finally:
                    self.pbar.close()
            
            def __enter__(self):
                return self
            
            def __exit__(self, exc_type, exc_val, exc_tb):
                self.pbar.close()
        
        return ProgressWrapper(iterable, pbar)
    else:
        return ProgressBar(total=total, **kwargs)

@contextmanager
def progress_context(total=None, **kwargs):
    pbar = ProgressBar(total=total, **kwargs)
    try:
        yield pbar
    finally:
        pbar.close()

tqdm = progress_bar
