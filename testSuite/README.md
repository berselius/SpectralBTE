# Running the test problems

Eventually I will have a script that does this and/or cmake will do it. But until then, this is what you need to do

* Copy all the `*.test.*` files into your `input/` directory

`cp *.test.* ../input`

* For each of the tests run

`boltz_ <testname>.in <testname>.out`

* This will produce data in `Data/`
* For each of the `.base` files in `testSuite`, do a diff between the output in `Data` and the base case, e.g.

`diff Data/rho_BKW16.in_BKW16.out_default.plt testSuite/rho_BKW16.base`

