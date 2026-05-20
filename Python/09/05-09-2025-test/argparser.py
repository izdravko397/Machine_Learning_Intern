import sys

class ArgumentParser:
    def __init__(self, description=None):
        self.description = description
        self.optional_args = {('--help', '-h'): {'help': 'show this help message and exit'}}
        self.opt_arg_keys = []
        self.positional_args = {}
        self.pos_args_keys = []

    def add_argument(self, flag_or_posname: str, second_flag=None, /, action=None, type=None, default=None, choices=None, help=None):
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

            self.optional_args[(long_flag, short_flag)] = {}
            self.opt_arg_keys.append((long_flag, short_flag))
            arg_dict = self.optional_args[(long_flag, short_flag)]

        else:
            if pos_arg_name:
                self.positional_args[pos_arg_name] = {}
                self.pos_args_keys.append(pos_arg_name)
                arg_dict = self.positional_args[pos_arg_name]
            else:
                self.optional_args[(long_flag, short_flag)] = {}
                self.opt_arg_keys.append((long_flag, short_flag))
                arg_dict = self.optional_args[(long_flag, short_flag)]

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

    def _get_optional_arg_params(self, flag):
        flag_params = None

        for f, params in self.optional_args.items():
            if flag in f:
                flag_params = params
                break
        else:
            raise TypeError('invalid oprional arg')
        
        return flag_params, f
    
    def _get_optional_arg_name(self, long, short):
        return long[2:] if long else short[1:]

    def parse_args(self):
        class Parse_class:
            def __init__(self):
                pass

        pars_obj = Parse_class

        file_args = iter(sys.argv)
        fname = next(file_args)
        flag = None
        pos_inx = 0
        flag_params = None
        arg_flags = None
        name = None
        defined_optional_flags = set()

        for item in file_args:
            if item in {'-h', '--help'}:
                sys.stdout.write(self._make_help_msg(fname))
                exit()

            elif not item.startswith('-'):
                if flag:
                    if flag_params.get('action') in {"store_true", "store_false"}:
                        raise TypeError('flag agr cant get value')
                    
                    # type = flag_params.get('type')
                    if type := flag_params.get('type'):
                        try:
                            item = type(item)
                        except:
                            error = self._make_usage(fname)
                            error += f"{fname}: error: argument square: invalid {type.__name__} value: '{item}'"
                            print(error)
                            exit()

                    if choices := flag_params.get('choices'):
                        for i in choices:
                            if i == item:
                                break
                        else:
                            error = self._make_usage(fname)
                            error += f"{fname}: error: argument {flag}: invalid choice: {i + 1} (choose from {', '.join(str(i) for i in choices)})"
                            print(error)
                            exit() 
                    
                    setattr(pars_obj, name, item)
                    defined_optional_flags.add(flag)
                    flag = None
                    flag_params = None
                    arg_flags = None
                    name = None
                else:
                    if not self.positional_args:
                        raise TypeError('no positional args')
                    
                    arg_name = self.pos_args_keys[pos_inx]
                    
                    if type := self.positional_args[arg_name].get('type'):
                        try:
                            item = type(item)
                        except:
                            error = self._make_usage(fname)
                            error += f"{fname}: error: argument square: invalid {type.__name__} value: '{item}'"
                            print(error)
                            exit()

                    pos_inx += 1
                    setattr(pars_obj, arg_name, item)

            else:
                if flag:
                    raise TypeError(f'expected value for {flag}')
                
                flag = item
                flag_params, arg_flags = self._get_optional_arg_params(flag)
                name = self._get_optional_arg_name(arg_flags[0], arg_flags[1])
                action = flag_params.get('action')
                defined_optional_flags.add(flag)

                if action == "store_true":
                    setattr(pars_obj, name, True)
                    flag = None
                    
                elif action == "store_false":
                    setattr(pars_obj, name, False)
                    flag = None

        if flag:
            error = self._make_usage(fname)
            error += f'{fname}: error: argument {flag}: expected one argument'
            print(error)
            exit()

        if pos_inx < len(self.pos_args_keys):
            error = self._make_usage(fname)
            error += f'{fname}: error: the following arguments are required: {', '.join(arg for arg in self.pos_args_keys)}'
            print(error)
            exit()

        for long, short in self.opt_arg_keys:
            if long in defined_optional_flags or short in defined_optional_flags:
                continue

            flag_params, arg_flags = self._get_optional_arg_params(long if long else short)
            name = self._get_optional_arg_name(arg_flags[0], arg_flags[1])
            action = flag_params.get('action')

            default = flag_params.get('default')
            if default is not None:
                val = default

            elif action == 'store_true':
                val = False

            elif action == 'store_false':
                val = True
            else:
                val = 0

            setattr(pars_obj, name, val)
                
        return pars_obj
    
    def _make_usage(self, fname):
        res = f'usage: {fname} [-h]'

        for arg, val in self.optional_args.items():
            if arg[0] == '--help':
                continue
            res += f' [{arg[1]}{' '  + val['ph'] if val.get('ph') else ''}]'
        
        for arg in self.positional_args:
            res += f' {arg}'

        res += '\n\n'
        return res      

    def _make_help_msg(self, fname):
        res = self._make_usage(fname)

        if self.description:
            res += self.description + '\n\n'

        if self.positional_args:
            res += 'positional arguments:\n'
        for arg, val in self.positional_args.items():
            res += f'  {format(arg, '<20')}{val['help'] if val.get('help') else ''}\n'
        res += '\n'

        res += 'options:\n'
        for (long, short), val in self.optional_args.items():
            res += f'  {format(f'{short}, {long}{' ' + val['ph'] if val.get('ph') else ''}', '<20')}{val['help'] if val.get('help') else ''}\n'

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