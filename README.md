This gh-pages uses [Mkdocs Material](https://squidfunk.github.io/mkdocs-material/) and [mike](https://github.com/jimporter/mike).

To delete a version from the branch, clone the repository, setup hatch. And run the `mike delete` command using an identified from the `json` versions list.

```shell
hatch run docs:mike delete [identifier]...
```
