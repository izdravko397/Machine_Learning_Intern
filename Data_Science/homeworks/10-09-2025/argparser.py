import sys

class ArgumentParser:
    def __init__(self, description=None):
        self.description = description
        self.optional_args = {'--help': {'help': 'show this help message and exit'},
                                '-h': {'help': 'show this help message and exit'}}
        self.positional_args = {}
        self.fname = ''

    def add_argument(self, flag_or_posname: str, second_flag=None, /, 
                     action=None, type=None, default=None, choices=None, nargs=None, help=None):
        long_flag = ''
        short_flag = ''
        pos_arg_name = ''

        if flag_or_posname.startswith('--'):
            long_flag = flag_or_posname
        elif flag_or_posname.startswith('-'):
            short_flag = flag_or_posname
        else:
            pos_arg_name = flag_or_posname
        
        if second_flag is not None:
            if pos_arg_name:
                raise TypeError
            
            if not second_flag.startswith('-'):
                raise TypeError
            
            if not long_flag:
                long_flag = second_flag
            else:
                short_flag = second_flag

            self.optional_args[long_flag] = {}
            self.optional_args[short_flag] = {}

        else:
            if pos_arg_name:
                self.positional_args[pos_arg_name] = {}
            elif long_flag:
                self.optional_args[long_flag] = {}
            else:
                self.optional_args[short_flag] = {}

        arg_dict = {}

        if action in {"store_true", "store_false"}:
            if any(a != None for a in [type, default]):
                raise TypeError("argument: 'type' or 'default' not allowed with store_true/store_false")
        else:
            arg_dict['ph'] = str(set(choices)) if choices else long_flag[2:].upper()
            
        if action is not None:
            arg_dict['action'] = action

        if type is not None:
            arg_dict['type'] = type

        if choices is not None:
            arg_dict['choices'] = choices

        if default is not None:
            arg_dict['default'] = default

        if help is not None:
            arg_dict['help'] = help

        arg_dict['nargs'] = nargs

        if short_flag:
            arg_dict['name'] = long_flag[2:] if long_flag else short_flag[1:]
            self.optional_args[short_flag] = arg_dict

        if long_flag:
            arg_dict['name'] = long_flag[2:]
            self.optional_args[long_flag] = arg_dict

        if pos_arg_name:
            arg_dict['name'] = pos_arg_name
            self.positional_args[pos_arg_name] = arg_dict

        arg_dict['name'] = arg_dict['name'].replace('-', '_')

    def _nargs(self, arguments, val):
        res = []

        if val == '*':
            n = sys.maxsize
        elif val == '?':
            n = 1
        else:
            n = val

        for _ in range(n):
            try:
                item = next(arguments)
                if item.startswith('-'):
                    raise Exception

                res.append(item)
            except:
                if isinstance(val, int) and len(res) != val:
                    self._print_error()
                break

        return res
    
    def _print_error(self, msg):
        error = self._make_usage() + msg
        print(error)
        exit()

    def parse_args(self):
        class Parse_class:
            pass

        pars_obj = Parse_class()
        positional_args = iter(self.positional_args.items())
        arguments = iter(sys.argv)
        self.fname = next(arguments)
        flag = None
        flag_params = None
        name = None

        for item in arguments:
            if item in {'-h', '--help'}:
                sys.stdout.write(self._make_help_msg())
                exit()

            elif not item.startswith('-'): # For value
                if flag: # Flag before value
                    if type := flag_params.get('type'):
                        try:
                            item = type(item)
                        except:
                            msg = f"{self.fname}: error: argument square: invalid {type.__name__} value: '{item}'"
                            self._print_error(msg)

                    if choices := flag_params.get('choices'):
                        for i in choices:
                            if i == item:
                                break
                        else:
                            msg = f"{self.fname}: error: argument {flag}: invalid choice: {i + 1} (choose from {', '.join(str(i) for i in choices)})"
                            self._print_error(msg)
                    
                    setattr(pars_obj, name, item)
                    flag = None
                    flag_params = None
                    name = None
                else: # Positional Args
                    try:
                        arg_name, arg_params = next(positional_args)
                    except StopIteration:
                        raise TypeError('no positional args')
                    
                    if type := arg_params.get('type'):
                        try:
                            item = type(item)
                        except:
                            msg = f"{self.fname}: error: argument square: invalid {type.__name__} value: '{item}'"
                            self._print_error(msg)

                    if narg := arg_params.get('nargs'):
                        first_item = item
                        val_list = self._nargs(arguments, narg)
                        val_list.append(first_item)
                        setattr(pars_obj, arg_name, val_list)
                        continue

                    setattr(pars_obj, arg_name, item)

            else: # For flag
                if flag:
                    raise TypeError(f'expected value for {flag}')
                
                flag = item
                if not (flag_params := self.optional_args.get(item)):
                    self._print_error(f'Flag not exist {item}')

                name = flag_params.get('name')
                if narg := flag_params.get('nargs'):
                    val_list = self._nargs(arguments, narg)
                    setattr(pars_obj, name, val_list)
                    flag = None

                action = flag_params.get('action')
                if action == "store_true":
                    setattr(pars_obj, name, True)
                    flag = None
                elif action == "store_false":
                    setattr(pars_obj, name, False)
                    flag = None

        if flag:
            msg = f'{self.fname}: error: argument {flag}: expected one argument'
            self._print_error(msg)

        for a, p in positional_args:
            if narg := p.get('nargs'):
                val_list = self._nargs(arguments, narg)
                setattr(pars_obj, a, val_list)
            else:
                msg = f'{self.fname}: error: the following arguments are required: {', '.join(arg for arg in self.positional_args)}'
                self._print_error(msg)

        for flag, params in self.optional_args.items():
            if flag in ("-h", "--help"):
                continue

            name = params.get('name')
            if hasattr(pars_obj, name):
                continue

            action = params.get("action")
            default = params.get("default")

            if default is not None:
                val = default
            elif action == "store_true":
                val = False
            elif action == "store_false":
                val = True
            else:
                val = None

            setattr(pars_obj, name, val)

        return pars_obj
    
    def _make_usage(self):
        res = f'usage: {self.fname} [-h]'

        for arg, val in self.optional_args.items():
            if arg == '--help' or arg == '-h':
                continue
            res += f' [{arg}{' '  + val['ph'] if val.get('ph') else ''}]'
        
        for arg in self.positional_args:
            res += f' {arg}'

        res += '\n\n'
        return res      

    def _make_help_msg(self):
        res = self._make_usage()

        if self.description:
            res += self.description + '\n\n'

        if self.positional_args:
            res += 'positional arguments:\n'
        for arg, val in self.positional_args.items():
            res += f'  {format(arg, '<20')}{val['help'] if val.get('help') else ''}\n'
        res += '\n'

        res += 'options:\n'
        for flag, val in self.optional_args.items():
            res += f'  {format(f'{flag}{' ' + val['ph'] if val.get('ph') else ''}', '<20')}{val['help'] if val.get('help') else ''}\n'

        return res

# parser = ArgumentParser(description='MyParser')
# # parser.add_argument("square", help="display a square of a given number", type=int)

# parser.add_argument("--verbosity", type=int, default=0, help="increase output verbosity")
# args = parser.parse_args()
# # print(args.square**2)

# if args.verbosity:
#     print("verbosity turned on")
#================================================
# parser = ArgumentParser()
# parser.add_argument("-v", "--verbose", help="increase output verbosity",
#                     action="store_true")
# args = parser.parse_args()
# if args.verbose:
#     print("verbosity turned on")
#=======================================================

# parser = ArgumentParser()
# parser.add_argument("square", type=int,
#                     help="display a square of a given number")
# parser.add_argument("-v", "--verbose", action="store_true",
#                     help="increase output verbosity")
# args = parser.parse_args()
# answer = args.square**2
# if args.verbose:
#     print(f"the square of {args.square} equals {answer}")
# else:
#     print(answer)
#======================================================
# parser = ArgumentParser()
# parser.add_argument("square", type=int,
#                     help="display a square of a given number")

# parser.add_argument("-v", "--verbosity", type=int,
#                     help="increase output verbosity")

# args = parser.parse_args()
# answer = args.square**2

# if args.verbosity == 2:
#     print(f"the square of {args.square} equals {answer}")
# elif args.verbosity == 1:
#     print(f"{args.square}^2 == {answer}")
# else:
#     print(answer)

#============================================================

# parser = ArgumentParser()
# parser.add_argument("square", type=int,
#                     help="display a square of a given number")
# parser.add_argument("-v", "--verbosity", type=int, choices=[0, 1, 2],
#                     help="increase output verbosity")
# args = parser.parse_args()
# answer = args.square**2

# if args.verbosity == 2:
#     print(f"the square of {args.square} equals {answer}")
# elif args.verbosity == 1:
#     print(f"{args.square}^2 == {answer}")
# else:
#     print(answer)

#==================================================
# parser = ArgumentParser()
# parser.add_argument("square", type=int,
#                     help="display the square of a given number")
# parser.add_argument("-v", "--verbosity", action="count",
#                     help="increase output verbosity")
# args = parser.parse_args()

# answer = args.square**2
# if args.verbosity == 2:
#     print(f"the square of {args.square} equals {answer}")
# elif args.verbosity == 1:
#     print(f"{args.square}^2 == {answer}")
# else:
#     print(answer)