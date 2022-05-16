# -*- coding: utf-8 -*-


def getFileContent(file_name):
    result = []
    with open(file_name, 'r', encoding='utf-8') as fp:
        rows = fp.readlines()
        for row in rows:
            try:
                row = row.strip(' \n').split(' ')
                info = {'file': row[0]}
                info['points'] = []
                for i in range(15, len(row), 2):
                    info['points'].append((float(row[i]), float(row[i+1])))
                result.append(info)
            except IndexError:
                print('IndexError')
            except ValueError:
                print('ValueError')
    return result


def parseResult(file_content):
    result = []
    for f in file_content:
        line_result = ['version: 1', 'n_points:  68', '{']
        for row in f['points']:
            line_result.append('%f %f' % row)
        line_result.append('}')
        result.append({'image': f['file'], 'content': '\n'.join(line_result)})
    return result


def saveResult(result):
    for r in result:
        name = r['image'].split('.')[:-1]
        name.append('pts')
        name = '.'.join(name)
        content = r['content']
        with open(name, 'w+', encoding='utf-8') as fp:
            fp.write(content)


def main():
    r = parseResult(getFileContent('MultiPIE_semifrontal_train.txt'))
    saveResult(r)
    print(r)


if '__main__' == __name__:
    main()

