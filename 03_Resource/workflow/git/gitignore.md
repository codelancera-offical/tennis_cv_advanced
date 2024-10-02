在 Git 项目中添加 `.gitignore` 文件的步骤如下：

1. **创建 `.gitignore` 文件**：
   - 在项目的根目录下创建一个名为 `.gitignore` 的文件。
   - 你可以使用命令行或文本编辑器来创建，例如：
     ```bash
     touch .gitignore
     ```

2. **编辑 `.gitignore` 文件**：
   - 打开 `.gitignore` 文件，并在其中添加你希望 Git 忽略的文件或目录。每行一个文件或目录。例如：
     ```
     # 忽略所有 .log 文件
     *.log
     
     # 忽略 node_modules 目录
     node_modules/
     
     # 忽略编译生成的文件
     *.o
     *.a
     *.so
     ```

3. **提交 `.gitignore` 文件**：
   - 保存 `.gitignore` 文件后，将其添加到 Git 版本控制中并提交：
     ```bash
     git add .gitignore
     git commit -m "添加 .gitignore 文件"
     ```

4. **查看 `.gitignore` 是否生效**：
   - 如果你已经将某些文件提交到 Git 仓库中，`.gitignore` 只能忽略新的文件或修改未被跟踪的文件。你可以使用以下命令查看 Git 是否忽略了预期的文件：
     ```bash
     git status
     ```

如果有已经被提交的文件需要被忽略，你可以先将它们从 Git 缓存中删除，然后再更新 `.gitignore` 文件。