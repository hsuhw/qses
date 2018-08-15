import os
import re
import shutil

from glob import glob
from pybuilder.core import init, before, use_plugin
from pybuilder.utils import assert_can_execute

use_plugin('python.core')
use_plugin('python.unittest')
use_plugin('python.flake8')
use_plugin('python.pylint')
use_plugin('pypi:pybuilder_pylint_extended')

default_task = ['clean', 'publish']


@init
def set_properties(project):
    project.set_property('verbose', True)
    project.set_property('dir_source_main_antlr', 'src/main/antlr')
    project.set_property('dir_source_antlr_dest', 'src/main/python/generated')
    project.set_property('antlr_generated_types',
                         ['*.py', '*.interp', '*.tokens'])

    # flake8
    project.set_property('flake8_break_build', True)
    project.set_property('flake8_include_scripts', True)
    project.set_property('flake8_include_test_sources', True)
    project.set_property('flake8_max_line_length', 80)
    project.set_property('flake8_verbose_output', True)

    # pylint
    project.set_property('pylint_break_build', True)
    project.set_property('pylint_include_scripts', True)
    project.set_property('pylint_include_test_sources', True)


@before('compile_sources')
def compile_antlr_grammar_sources(project, logger):
    clean_antlr_generated_files(project, logger)
    assert_can_execute(['antlr4'], 'antlr4', 'compile_antlr_grammar_sources')

    src_dir = project.get_property('dir_source_main_antlr')
    grammar_items = ['SMTLIB26Lexer.g4', 'SMTLIB26Parser.g4']
    grammar_files = []
    for item in grammar_items:
        grammar_files.append(f'{src_dir}/{item}')
    cmd = f"antlr4 -Dlanguage=Python3 {' '.join(grammar_files)}"

    import subprocess
    subprocess.check_output(cmd, shell=True)

    dest_dir = project.get_property('dir_source_antlr_dest')
    tgt_files = []
    for extension in project.get_property('antlr_generated_types'):
        tgt_files.extend(glob(f'{src_dir}/SMTLIB26{extension}'))
    for file in tgt_files:
        if re.match('.*\.tokens', file):
            shutil.copy(file, dest_dir)
        else:
            shutil.move(file, dest_dir)


@before('clean')
def clean_antlr_generated_files(project, logger):
    dest_dir = project.get_property('dir_source_antlr_dest')
    tgt_files = []
    for extension in project.get_property('antlr_generated_types'):
        tgt_files.extend(glob(f'{dest_dir}/SMTLIB26{extension}'))
    for file in tgt_files:
        logger.info(f'Removing generated file {file}')
        os.remove(file)
