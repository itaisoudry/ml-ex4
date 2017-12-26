import os

inputDir = '/home/soudry/Desktop/input/'
outputDir = '/home/soudry/Desktop/output2/'
resDir = '/home/soudry/Desktop/res/'
compilerPath = '/home/soudry/Compilation/EX3/COMPILER'
logFilePath = ''

directory = os.fsencode(inputDir)
log = open('log.txt', 'w')
for file in os.listdir(directory):
    filename = os.fsdecode(file)
    os.system('''java -jar {} {}{} {}{}'''.format(
        compilerPath, inputDir, filename, outputDir, filename))

for root, dirs, filenames in os.walk(outputDir):
    for f in filenames:

        if f.startswith('TEST'):
            for root2, dirs2, filenames2 in os.walk(resDir):
                log.write('for1\n')
                for f2 in filenames2:
                    log.write('for2\n')
                    if f2.startswith(f[:-4]):
                        log.write('inside\n')
                        with open(os.path.join(root2, f2), 'r') as fin:
                            with open(os.path.join(root, f), 'r') as fin2:
                                acutal = fin2.read()
                                expected = fin.read()
                                if expected in acutal:
                                    log.write('PASSED - ' + f)
                                else:
                                    log.write('FAILED - ' + f)
                                    log.write('EXPECTED: ' + expected + 'ACTUAL: ' + acutal + '\n')
                        log.write("=======================================\n")
                        break

log.close()
