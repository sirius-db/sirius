#!/usr/bin/env python3
# Copyright 2026, Sirius Contributors. SPDX-License-Identifier: Apache-2.0
"""Run under the frozen Pixi environment. Outputs must be new directories."""
import argparse, hashlib, json, os, shlex, shutil, subprocess, time
from pathlib import Path
from instrument import instrument
from archive_audit import verify
import xml.etree.ElementTree as ET

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def git(r, *args):
    return subprocess.check_output(['git', '-C', str(r), *args], text=True).strip()

def compile_command(b, original, source, obj, include):
    entry = next((x for x in json.loads((b / 'compile_commands.json').read_text()) if Path(x['file']) == original))
    a = shlex.split(entry['command'])
    cmd = []
    i = 0
    while i < len(a):
        x = a[i]
        if x in ['-o', '-MF', '-MT', '-MQ']:
            i += 2
            continue
        if x in ['-MD', '-MMD'] or Path(x).name == 'sccache':
            i += 1
            continue
        cmd.append(str(source) if x == str(original) else x)
        i += 1
    return (cmd + ['-I', str(include), '-o', str(obj)], Path(entry['directory']))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--reference', type=Path, required=True)
    p.add_argument('--candidate', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args()
    source_tool = Path(__file__).resolve().parent
    args.out.mkdir(exist_ok=False, parents=True)
    tool = args.out / 'tools'
    shutil.copytree(source_tool, tool, ignore=shutil.ignore_patterns('__pycache__'))
    for (label, r) in [('reference', args.reference), ('candidate', args.candidate)]:
        d = args.out / label
        d.mkdir()
        b = r / 'build/release'
        original = r / 'src/io/rest/rest_reactor.cpp'
        (source, edits, inverse) = instrument(original.read_text())
        assert inverse == original.read_text()
        (d / 'rest_reactor.cpp').write_text(source)
        (d / 'insertions.json').write_text(json.dumps(edits, indent=2))
        fixture = r / 'test/cpp/io/rest/test_rest_ioctx_integration.cpp'
        (d / 'fixture.cpp').write_text(fixture.read_text() + '\n' + (tool / 'fixture_extension.cpp').read_text())
        m = {'engine_root': str(r), 'head': git(r, 'rev-parse', 'HEAD'), 'tree': git(r, 'rev-parse', 'HEAD^{tree}'), 'status': git(r, 'status', '--porcelain'), 'lock_sha256': sha(r / 'pixi.lock'), 'core_sha256': sha(b / 'libsirius_core.a'), 'unittest_sha256': sha(b / 'test/cpp/sirius_unittest'), 'source_sha256': sha(original), 'fixture_sha256': sha(fixture), 'generated_source_sha256': sha(d / 'rest_reactor.cpp'), 'generated_fixture_sha256': sha(d / 'fixture.cpp'), 'tools': {f.name: sha(f) for f in sorted(tool.iterdir()) if f.is_file()}, 'inverse_exact': True, 'steps': []}
        env = os.environ.copy()
        env.pop('SIRIUS_TEST_REST_REQUEST_LOG', None)
        env['SIRIUS_TEST_S3_AUTO'] = '1'
        env['SIRIUS_TEST_S3_STRICT'] = '1'
        env['D1_OUT'] = str(d)

        def run(name, cmd, cwd):
            t = time.time()
            with (d / (name + '.stdout')).open('w') as o, (d / (name + '.stderr')).open('w') as e:
                result = subprocess.run(cmd, cwd=cwd, stdout=o, stderr=e, env=env, timeout=600)
            m['steps'].append({'name': name, 'command': cmd, 'cwd': str(cwd), 'exit': result.returncode, 'seconds': time.time() - t})
            (d / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
            print(label, name, result.returncode, flush=True)
            result.check_returncode()
        members = subprocess.check_output(['ar', 't', str(b / 'libsirius_core.a')], text=True).splitlines()
        matching = [x for x in members if x.endswith('rest_reactor.cpp.o')]
        assert len(matching) == 1
        obj = d / matching[0]
        (cmd, cwd) = compile_command(b, original, d / 'rest_reactor.cpp', obj, tool)
        run('compile-reactor', cmd, cwd)
        shutil.copyfile(b / 'libsirius_core.a', d / 'libsirius_core.a')
        run('replace-one-member', ['ar', 'r', str(d / 'libsirius_core.a'), str(obj)], d)
        after = subprocess.check_output(['ar', 't', str(d / 'libsirius_core.a')], text=True).splitlines()
        assert members == after
        m['archive_audit'] = verify(b / 'libsirius_core.a', d / 'libsirius_core.a')
        (cmd, cwd) = compile_command(b, fixture, d / 'fixture.cpp', d / 'fixture.o', tool)
        run('compile-fixture', cmd, cwd)
        line = subprocess.check_output([str(r / '.pixi/envs/default/bin/ninja'), '-C', str(b), '-t', 'commands', 'sirius_unittest'], text=True).splitlines()[-1]
        assert line.startswith(': && ') and line.endswith(' && :')
        link = shlex.split(line[5:-5])
        oldobj = 'CMakeFiles/sirius_unittest.dir/test/cpp/io/rest/test_rest_ioctx_integration.cpp.o'
        assert link.count(oldobj) == 1
        cores = [x for x in link if x.endswith('/libsirius_core.a') or x == 'libsirius_core.a']
        assert len(cores) == 1
        link = [str(d / 'fixture.o') if x == oldobj else str(d / 'libsirius_core.a') if x == cores[0] else x for x in link]
        binary = d / 'd1_tests'
        link[link.index('-o') + 1] = str(binary)
        run('link', link, b)
        m['instrumented_core_sha256'] = sha(d / 'libsirius_core.a')
        m['binary_sha256'] = sha(binary)
        run('joint', [str(binary), '[d1_joint]', '--order', 'decl', '-r', 'xml::out=' + str(d / 'execution.xml')], r)
        execution = ET.parse(d / 'execution.xml').getroot()
        assert execution.find('OverallResultsCases').get('successes') == '1'
        assert execution.find('OverallResults').get('failures') == '0'
        sections = execution.findall('.//Section')
        assert len(sections) == 9
        assert all((x.find('OverallResults').get('failures') == '0' for x in sections))
        run('inherited-rest', [str(binary), '[rest]', '--order', 'decl', '-r', 'xml::out=' + str(d / 'inherited-rest.xml')], r)
        inherited = ET.parse(d / 'inherited-rest.xml').getroot()
        assert inherited.find('OverallResultsCases').get('successes') == '132'
        assert inherited.find('OverallResults').get('failures') == '0'
        assert len(inherited.findall('TestCase')) == 132
        assert sha(b / 'libsirius_core.a') == m['core_sha256']
        assert sha(b / 'test/cpp/sirius_unittest') == m['unittest_sha256']
        m['frozen_originals_unchanged'] = True
        (d / 'manifest.json').write_text(json.dumps(m, indent=2) + '\n')
    checks = [('qualify', [os.sys.executable, str(tool / 'validate.py'), str(args.out / 'reference'), str(args.out / 'candidate'), '--out', str(args.out / 'comparison.json')]), *[('negative-' + engine, [os.sys.executable, str(tool / 'test_validate.py'), str(args.out / engine)]) for engine in ('reference', 'candidate')]]
    receipts = []
    for (name, command) in checks:
        with (args.out / (name + '.stdout')).open('w') as stdout, (args.out / (name + '.stderr')).open('w') as stderr:
            result = subprocess.run(command, stdout=stdout, stderr=stderr, timeout=60)
        receipts.append({'name': name, 'command': command, 'exit': result.returncode})
        (args.out / 'qualification-commands.json').write_text(json.dumps(receipts, indent=2) + '\n')
        print(name, result.returncode, flush=True)
        result.check_returncode()
if __name__ == '__main__':
    main()
