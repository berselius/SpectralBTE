#!/bin/bash

grep x input > output
diff output target

exit $?
