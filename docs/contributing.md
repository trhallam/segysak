# Contributing

**Thank you for considering contributing to our project!**

This is a community-driven project, so it's people like you that make it useful and
successful.
These are some of the many ways to contribute:

* Submitting bug reports and feature requests
* Writing tutorials or examples
* Fixing typos and improving the documentation
* Writing code for everyone to use

If you get stuck at any point you can create an issue on GitHub (look for the *Issues*
tab in the repository) or contact us at one of the other channels mentioned below.

For more information on contributing to open source projects,
[GitHub's own guide](https://guides.github.com/activities/contributing-to-open-source/)
is a great starting point if you are new to version control.

Also, checkout the
[Zen of Scientific Software Maintenance](https://jrleeman.github.io/ScientificSoftwareMaintenance/)
for some guiding principles on how to create high quality scientific software
contributions.

## Ground Rules

The goal is to maintain a diverse community that's pleasant for everyone.
**Please be considerate and respectful of others**.
Everyone must abide by our [Code of Conduct](https://github.com/trhallam/segysak/blob/main/code_of_conduct.md) and we encourage all to read it carefully.

## What Can I Do?

* Tackle any issue that you wish! Some issues are labelled as **"good first issues"** to
  indicate that they are beginner friendly, meaning that they don't require extensive
  knowledge of the project.
* Make a tutorial or example of how to do something.
* Provide feedback about how we can improve the project or about your particular use
  case.
* Contribute code you already have. It doesn't need to be perfect! We will help you
  clean things up, test it, etc.

## How Can I Talk to You?

Discussion often happens in the repository [issues](https://github.com/trhallam/segysak/issues) tracker and individual pull requests.
For questions and general chat, please use the [Discussions](https://github.com/trhallam/segysak/discussions) board.

## Reporting a Bug

Find the *Issues* [tab](https://github.com/trhallam/segysak/issues) on the top of the Github repository and click *New Issue*.
You'll be prompted to choose between different types of issue, like bug reports and
feature requests.
Choose the one that best matches your need.
The Issue will be populated with one of our templates.
**Please try to fillout the template with as much detail as you can**.

!!! tip

    Remember: the more information we have, the easier it will be for us to solve your
    problem.

## Editing the Documentation

If you're browsing the documentation and notice a typo or something that could be
improved, please consider letting us know by creating an issue (see [reporting_a_bug](#reporting-a-bug))
or submitting a fix (even better).

You can submit fixes to the documentation pages completely online without having to
download and install anything:

* On each documentation page, there should be an "edit on Github" link at the very
  top.
* Click on that link to open the respective source file (usually an `.rst` file in the
  `doc` folder) on Github for editing online (you'll need a Github account).
* Make your desired changes.
* When you're done, scroll to the bottom of the page.
* Fill out the two fields under "Commit changes": the first is a short title describing
  your fixes; the second is a more detailed description of the changes. Try to be as
  detailed as possible and describe *why* you changed something.
* Click on the "Commit changes" button to open a
  [pull request (see below)](#pull-requests).
* We'll review your changes and then merge them in if everything is OK.
* Done

Alternatively, you can make the changes offline to the files in the `doc` folder or the
example scripts. See [Contributing Code](#contributing-code) for instructions.

## Contributing Code

**Is this your first contribution?**
Please take a look at these resources to learn about git and pull requests don't
hesitate to [ask questions](#how-can-i-talk-to-you):

* [How to Contribute to Open Source](https://opensource.guide/how-to-contribute/)
* [Aaron Meurer's tutorial on the git workflow](http://www.asmeurer.com/git-workflow/)
* [How to Contribute to an Open Source Project on GitHub](https://egghead.io/courses/how-to-contribute-to-an-open-source-project-on-github)

### General guidelines

We follow the [git pull request workflow](https://en.wikipedia.org/wiki/Continuous_integration) 
to make changes to our codebase.
Every change made goes through a pull request, even our own, so that our
[continuous integration](http://www.asmeurer.com/git-workflow/) services
have a change to check that the code is up to standards and passes all our tests.
This way, the *master* branch is always stable.

General guidelines for pull requests (PRs):

  * **Open an issue first** describing what you want to do. If there is already an issue
    that matches your PR, leave a comment there instead to let us know what you plan to
    do.
  * Each pull request should consist of a **small** and logical collection of changes.
  * Larger changes should be broken down into smaller components and integrated
    separately.
  * Bug fixes should be submitted in separate PRs.
  * Describe what your PR changes and *why* this is a good thing. Be as specific as you
    can. The PR description is how we keep track of the changes made to the project over
    time.
  * Do not commit changes to files that are irrelevant to your feature or bugfix (eg:
    `.gitignore`, IDE project files, etc).
  * Write descriptive commit messages. Chris Beams has written a
    [guide](https://cbea.ms/git-commit/) on how to write good commit
    messages.
  * Be willing to accept criticism and work on improving your code; we don't want to break
    other users' code, so care must be taken not to introduce bugs.
  * Be aware that the pull request review process is not immediate, and is generally
    proportional to the size of the pull request.


### Setting up your environment

This project uses the [Hatch](https://hatch.pypa.io/) python project tool.
Hatch takes care of Python versions and dependencies (as defined in the `pyproject.toml`) and development
environments.

Follow the [hatch instructions](https://hatch.pypa.io/dev/install/) for installing `hatch` 
on your system.

To run tests on your system in the default Python version

```bash
hatch test
```

This installs the virtual environment, installs the project in development mode and runs the test suite.
Under the hood the command uses the `pytest` package. If you need to modify the testing you can use the
`hatch-test` environment and `pytest` command directly.

```bash
hatch run hatch-test.py3.12:pytest --help
```

### Code style

We use [Black](https://github.com/ambv/black) to format the code so we don't have to
think about it. Black loosely follows the [PEP8](http://pep8.org) guide but with a few differences.
Regardless, you won't have to worry about formatting the code yourself.

Don't worry if you forget to do it. Our continuous integration systems will warn us and you can make 
a new commit with the formatted code.

We also use [flake8](http://flake8.pycqa.org/en/latest/) and 
[pylint](https://www.pylint.org/) to check the quality of the code and quickly catch
common errors.

### Docstrings

**All docstrings** should follow the
[Google Style Guide](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings).
All functions/classes/methods should have docstrings with a full description of all
arguments and return values.

While the maximum line length for code is automatically set by *Black*, docstrings
must be formatted manually. To play nicely with Jupyter and IPython, **keep docstrings
limited to 79 characters** per line. We don't have a good way of enforcing this
automatically yet, so please do your best.

### Testing your code

Automated testing helps ensure that our code is as free of bugs as it can be.
It also lets us know immediately if a change we make breaks any other part of the code.

All of our test code and data are stored in the `tests` subpackage.
We use the [pytest](https://pytest.org/) framework to run the test suite.

Please write tests for your code so that we can be sure that it won't break any of the
existing functionality.
Tests also help us be confident that we won't break your code in the future.

If you're **new to testing**, see existing test files for examples of things to do.
**Don't let the tests keep you from submitting your contribution!**
If you're not sure how to do this or are having trouble, submit your pull request
anyway.
We will help you create the tests and sort out any kind of problem during code review.

Run the tests and calculate test coverage using:

```bash
hatch test
```

A coverage report can be generated with

```bash
hatch test -c
```

The coverage report will let you know which lines of code are touched by the tests.
**Strive to get 100% coverage for the lines you changed.**
It's OK if you can't or don't know how to test something.
Leave a comment in the PR and we'll help you out.

## Documentation

Most documentation sources are in the `docs` folder.
We use [Mkdocs](https://www.mkdocs.org/) to build the web pages from these sources.

To serve a live copy of the documentation locally, Mkdocs provides a local web-server and auto-building tool

```bash
hatch run docs:serve
```

To build the HTML files as a static site

```bash
hatch run docs:build
```

## Examples

Examples are written as Jupyter Notebooks and converted to `py` files using [Jupytext](https://github.com/mwouts/jupytext).
Each example is executed prior to building the documentation so that they are kept
relevant and up-to-date.

To contribute an example, start by writing a notebook that uses either the data in
the examples folder, or a small (less than 50Mb) dataset you can contribute to the project.

Then using Jupytext sync your notebook with a Percent style `py` file using the Jupytext menu.
This `py` file will form the basis of your pull request.

When the file has been checked, the example can be added to the documentation to ensure 
it is rendered in the final documentation output.

Other examples not suitable for the documentation are welcome and can be submitted as notebooks
(without execution) in the `examples/notebooks` folder of the repository.

You can start a Jupyter session in the project directory using hatch

```bash
hatch run docs:jupyter lab
```

## Code Review

After you've submitted a pull request, you should expect to hear at least a comment
within a couple of days.
We may suggest some changes or improvements or alternatives.

Some things that will increase the chance that your pull request is accepted quickly:

* Write a good and detailed description of what the PR does.
* Write tests for the code you wrote/modified.
* Readable code is better than clever code (even with comments).
* Write documentation for your code (docstrings) and leave comments explaining the
  *reason* behind non-obvious things.
* Include an example of new features in the gallery or tutorials.
* Follow the PEP8_ style guide for code and the
  [`Google Docstring Guide`](https://google.github.io/styleguide/pyguide.html#38-comments-and-docstrings)
  for documentation.

Pull requests will automatically have tests run by Github Actions.
This includes running both the unit tests as well as code linters.
Github will show the status of these checks on the pull request.
Try to get them all passing (green).
If you have any trouble, leave a comment in the PR or
[get in touch](#how-can-i-talk-to-you).

## Attribution

This contributing document is largely based upon the work by the Fatiando a Terra project.