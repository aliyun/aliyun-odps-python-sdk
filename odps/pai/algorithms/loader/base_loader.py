# encoding: utf-8
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.

import codecs
import getpass
import json
from inspect import getmembers
import os
import py_compile
import random
from string import Template
import tempfile
from six import PY2, iteritems
from six.moves import range

user_code_id = [1]

PARAMETER_ACCESSOR_BLACK_LIST = {'isInheritLastType', 'inputTableName', 'inputTablePartitions', 'outputTableName',
                                 'outputTablePartitions'}

TEMPLATE_IMPORT = Template("""
# encoding: utf-8
from odps.pai.algorithms.base_algo import $base_class
from odps.pai.algorithms.algorithm_nodes import AlgorithmParameter, AlgorithmIO
""")

TEMPLATE_CLASS = Template("""
class $class_name($base_class):
    def __init__(self, **kwargs):
        parameters = {
$param_defs
        }
        ports = {
$port_defs
        }
        metas = {
$metas
        }
        super($class_name, self).__init__("$code_name", parameters, ports, metas)
$accessors
""")

TEMPLATE_PARAMETER = Template("""
            "$name": self._fill_param(AlgorithmParameter("$name", value=$value, input_name=$input_name,
                output_name=$output_name, exporter=$exporter, exported=$exported), kwargs, "$method_name"),
""")

TEMPLATE_PORT = Template("""
            "$name": AlgorithmIO("$name", $seq, "$direction", port_type="$port_type"),
""")

TEMPLATE_PARAMETER_ACCESSOR = Template("""
    def get_$method_name(self):
        return self._parameters["$name"].value

    def set_$method_name(self, value):
        self._parameters["$name"].value = value
        return self
""")


class BaseContentLoader(object):
    def __init__(self):
        pass

    @staticmethod
    def _generate_script_in_array(json_array, base_class):
        def convert_class_name(original):
            return "".join([s[0].upper() + s[1:len(s)] for s in original.split('_')])

        def convert_member_name(original):
            capitalized = "".join([s[0].upper() + s[1:len(s)] for s in original.split('_')])
            if capitalized.startswith('Is'):
                capitalized = capitalized[2:len(capitalized)]

            def to_member_name():
                for idx in range(len(capitalized)):
                    if capitalized[idx].isupper():
                        if idx == 0:
                            yield capitalized[0].lower()
                        elif idx < len(capitalized) - 1 and capitalized[idx + 1].islower():
                            yield '_' + capitalized[idx].lower()
                        elif idx > 0 and capitalized[idx - 1].islower():
                            yield '_' + capitalized[idx].lower()
                        else:
                            yield capitalized[idx].lower()
                    else:
                        yield capitalized[idx]
            return ''.join(to_member_name())

        def convert_parameters(params):
            for param in params:
                if 'value' not in param:
                    continue

                if param['type'] == 'string':
                    default = json.dumps(param['value']) if param['value'] != '' else 'None'
                elif param['type'] == 'boolean':
                    if param['value'].strip().lower() == 'true':
                        default = 'True'
                    else:
                        default = 'False'
                else:
                    if param['value'].isdecimal():
                        default = param['value']
                    else:
                        default = "'%s'" % param['value'] if param['value'] != '' else 'None'
                        param['type'] = 'string'
                input_name = "'%s'" % param['inputName'] if 'inputName' in param else 'None'
                output_name = "'%s'" % param['outputName'] if 'outputName' in param else 'None'
                exporter = "'%s'" % param['exporter'] if 'exporter' in param else 'None'
                exported = repr(param['exported'])
                method_name = convert_member_name(param['name'])
                yield TEMPLATE_PARAMETER.substitute(name=param['name'], value=default, input_name=input_name,
                                                    output_name=output_name, exporter=exporter, exported=exported,
                                                    method_name=method_name).strip('\n')

        def convert_ports(ports):
            for port in ports:
                yield TEMPLATE_PORT.substitute(name=port['name'], seq=port['sequence'], direction=port['ioType'],
                                               port_type=port['type']).strip('\n')

        def convert_parameter_setters(params):
            for param in params:
                if param['name'] in PARAMETER_ACCESSOR_BLACK_LIST:
                    continue
                yield TEMPLATE_PARAMETER_ACCESSOR.substitute(name=param['name'],
                                                             method_name=convert_member_name(param['name'])).strip('\n')

        def convert_node_defs():
            for nodeDef in json_array:
                code_name = nodeDef['codeName']

                parameter_defs = '\n'.join(convert_parameters(nodeDef['params']))
                port_defs = '\n'.join(convert_ports(nodeDef['ports']))
                metas = ','.join(['"%s": "%s"' % (k, v) for k, v in iteritems(nodeDef['metas'])])
                parameter_setters = '\n' + '\n\n'.join(convert_parameter_setters(nodeDef['params']))

                class_code = TEMPLATE_CLASS.substitute(accessors=parameter_setters, param_defs=parameter_defs,
                                                       port_defs=port_defs, metas=metas, code_name=code_name,
                                                       class_name=convert_class_name(code_name), base_class=base_class)
                yield class_code.strip('\n')

        return TEMPLATE_IMPORT.substitute(base_class=base_class) \
             .strip('\n') + '\n\n\n' + '\n\n\n'.join(convert_node_defs())

    @staticmethod
    def _load_source_module(code):
        script_name = tempfile.gettempdir() + os.sep + getpass.getuser() + "-" + str(os.getpid()) + "-" + \
                      str(random.randint(0, 99999))
        script_file = codecs.open(script_name + ".py", 'w', 'UTF-8')
        script_file.write(code)
        script_file.close()

        py_compile.compile(script_name + '.py', script_name + '.pyc')
        os.unlink(script_name + '.py')

        mod = import_compiled_module(script_name + '.pyc')
        os.unlink(script_name + '.pyc')

        return mod

    @staticmethod
    def _load_algorithms(json_array, base_class, env):
        code = BaseContentLoader._generate_script_in_array(json_array, base_class)
        new_module = BaseContentLoader._load_source_module(code)
        for name, dtype in getmembers(new_module):
            setattr(env, name, dtype)


def import_compiled_module(file_name):
    if PY2:
        import imp
        module = imp.load_compiled('user_code%d' % user_code_id[0], file_name)
    else:
        import importlib
        module = importlib.machinery.SourcelessFileLoader('user_code%d' % user_code_id[0], file_name).load_module()
    user_code_id[0] += 1
    return module
