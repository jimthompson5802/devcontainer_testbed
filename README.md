# devcontainer_testbed
testing various VSCode devcontainer setups


## devcontainer configuration parameters
The `devcontainer.json` file is used to define the development environment for Visual Studio Code's Remote - Containers extension. As of my last update in January 2022, here are the available properties in `devcontainer.json` and an example use for each:

1. **name**: A user-friendly string to distinguish the container.
   ```json
   "name": "My Dev Container"
   ```

2. **dockerFile** or **image**: Use `dockerFile` to specify a path to a Dockerfile or `image` to use an existing Docker image.
   ```json
   "dockerFile": "Dockerfile",
   // or
   "image": "node:14"
   ```

3. **context**: Sets the build context for the Docker build.
   ```json
   "context": "."
   ```

4. **build**: Configuration related to building the Docker container.
   - **args**: A set of Docker build arguments as key-value pairs.
   - **dockerfile**: Alternative way to specify the Dockerfile.
   - **context**: Alternative way to specify the build context.
   - **target**: Specify a build target to use from your Dockerfile.
   - **cacheFrom**: An array of images to use for Docker's `--cache-from` option.

   ```json
   "build": {
      "dockerfile": "Dockerfile",
      "context": ".",
      "args": { "VARIANT": "14" },
      "target": "dev"
   }
   ```

5. **runArgs**: Arguments to pass to `docker run`.
   ```json
   "runArgs": ["--name", "my-container-name"]
   ```

6. **mounts**: Additional mount points.
   ```json
   "mounts": ["source=/var/run/docker.sock,target=/var/run/docker.sock,type=bind"]
   ```

7. **appPort**: Forward a port, or an array of ports, from the container to the host.
   ```json
   "appPort": [8080, "5000-5005"]
   ```

8. **extensions**: List of extensions to install inside the container.
   ```json
   "extensions": ["ms-python.python", "ms-vscode.go"]
   ```

9. **settings**: Default settings to apply when the container is opened.
   ```json
   "settings": {
      "terminal.integrated.shell.linux": "/bin/bash"
   }
   ```

10. **remoteEnv**: Environment variables to set in the container.
    ```json
    "remoteEnv": {
       "MY_VARIABLE": "value"
    }
    ```

11. **containerEnv**: Environment variables for the container (available during the build process).
    ```json
    "containerEnv": {
       "MY_BUILD_VARIABLE": "value"
    }
    ```

12. **remoteUser**: Override the user that VS Code runs as in the container.
    ```json
    "remoteUser": "vscode"
    ```

13. **workspaceFolder**: The folder where your project should be copied inside the container.
    ```json
    "workspaceFolder": "/workspace"
    ```

14. **workspaceMount**: Override the default local mount point for the workspace.
    ```json
    "workspaceMount": "source=${localWorkspaceFolder},target=/workspace,type=bind,consistency=cached"
    ```

15. **postCreateCommand**: Command string or array of command arguments to run after creating the container.
    ```json
    "postCreateCommand": "npm install"
    ```

16. **postStartCommand**: Command to run after starting the container.
    ```json
    "postStartCommand": "echo 'Container started!'"
    ```

17. **postAttachCommand**: Command to run after attaching to the container.
    ```json
    "postAttachCommand": "echo 'Attached to container!'"
    ```

18. **initializeCommand**: Command to run on the host before the container starts.
    ```json
    "initializeCommand": "echo 'Initializing on host before container starts!'"
    ```

19. **shutdownAction**: Control the behavior when the last VS Code window in a container is closed.
    ```json
    "shutdownAction": "none"
    ```

20. **runOptions**: Add properties for `docker run`.
    ```json
    "runOptions": ["--cap-add=SYS_PTRACE", "--security-opt", "seccomp=unconfined"]
    ```

21. **features**: Enables or disables certain features.
    ```json
    "features": {
       "github": "false"
    }
    ```

22. **service**: Defines a set of Docker compose services to start.
    ```json
    "service": ["db", "api"]
    ```

23. **shutdownAction**: What to do when closing the last VS Code window connected to this container.
    ```json
    "shutdownAction": "stopContainer"
    ```

24. **extensionKind**: Override the kind of an extension if you know it has been misclassified.
    ```json
    "extensionKind": {
       "ms-python.python": ["workspace"]
    }
    ```

25. **forwardPorts**: Ports to automatically forward when the container starts.
    ```json
    "forwardPorts": [4000, 4001]
    ```

26. **externalExtensions**: Extensions to install from a local `.vsix` file or download URL.
    ```json
    "externalExtensions": [{
       "extensionId": "ms-vscode.cpptools",
       "version": "1.1.3"
    }]
    ```

Note: The configuration for `devcontainer.json` can change as the feature is further developed by Microsoft. Always refer to the official documentation for the most up-to-date information.


## VSCode Special Variables
In Visual Studio Code, there are a number of predefined variables that you can use within certain configuration files, like `launch.json`, `tasks.json`, and `settings.json`. These variables can be helpful to abstract values based on the user's current environment or workspace.

Here's a list of these special variables as of my last update in January 2022:

1. **`${workspaceFolder}`**: The path of the folder opened in VS Code. If no folder is opened, it will be undefined.

2. **`${workspaceFolderBasename}`**: The name of the folder opened in VS Code without any slashes (/).

3. **`${file}`**: The current opened file.

4. **`${relativeFile}`**: The current opened file relative to `workspaceFolder`.

5. **`${relativeFileDirname}`**: The current opened file's dirname relative to `workspaceFolder`.

6. **`${fileBasename}`**: The current opened file's basename.

7. **`${fileBasenameNoExtension}`**: The current opened file's basename with no file extension.

8. **`${fileDirname}`**: The current opened file's dirname.

9. **`${fileExtname}`**: The current opened file's extension.

10. **`${cwd}`**: The task runner's current working directory on startup.

11. **`${lineNumber}`**: The current selected line number in the active file.

12. **`${selectedText}`**: The current selected text in the active file.

13. **`${execPath}`**: The location of the VS Code executable.

14. **`${defaultBuildTask}`**: The name of the default build task. If there's no default build task, it will be undefined.

15. **`${pathSeparator}`**: The OS-specific path separator (`/` on macOS and Linux, `\` on Windows).

16. **`${env:Name}`**: Allows you to access environment variables on your system. Replace `Name` with the name of the environment variable. For example, `${env:PATH}` would get the PATH environment variable.

17. **`${command:commandID}`**: Executes a command and uses its return value. You'd replace `commandID` with the ID of the command you wish to run.

18. **`${input:variableID}`**: Invokes an input variable provider.

19. **`${command:extension.command}`**: Allows you to retrieve values provided by extensions. Replace `extension.command` with the command provided by the extension.

20. **`${config:configurationName}`**: Retrieves a configuration setting from the user or workspace settings. Replace `configurationName` with the desired setting name.

21. **`${workspaceStorage}`**: Location for storing workspace specific data.

22. **`${globalStorage}`**: Location for storing global data.

In addition to these, there are some additional special variables used specifically within `devcontainer.json` for DevContainers, like `${localEnv:VARNAME}` or `${containerEnv:VARNAME}`, but the list above covers the general-use special variables for VS Code.

Keep in mind that the availability of some of these variables might depend on the context in which you are using them. Also, always refer to the official VS Code documentation or release notes for any recent additions or changes to these variables.

### Example usage of special variables
Visual Studio Code (VS Code) supports several variables that can be used in its configuration settings to reference various paths, making the settings more dynamic. As of my last update in January 2022, here's a list of these variables and examples of their usage:

1. **`${workspaceFolder}`**: The path of the folder opened in VS Code.
   ```json
   {
      "settings": {
         "eslint.options": {
            "configFile": "${workspaceFolder}/.eslintrc.js"
         }
      }
   }
   ```

2. **`${workspaceFolderBasename}`**: The name of the folder opened in VS Code without any slashes (`/`).
   ```json
   {
      "settings": {
         "terminal.integrated.cwd": "${workspaceFolderBasename}"
      }
   }
   ```

3. **`${file}`**: The current opened file.
   ```json
   {
      "settings": {
         "editor.snippetSuggestions": "${file}"
      }
   }
   ```

4. **`${relativeFile}`**: The current opened file relative to `workspaceFolder`.
   ```json
   {
      "tasks": {
         "label": "echo",
         "type": "shell",
         "command": "echo ${relativeFile}",
         "problemMatcher": []
      }
   }
   ```

5. **`${relativeFileDirname}`**: The current opened file's directory path relative to `workspaceFolder`.
   ```json
   {
      "settings": {
         "terminal.integrated.cwd": "${relativeFileDirname}"
      }
   }
   ```

6. **`${fileBasename}`**: The current opened file's basename.
   ```json
   {
      "tasks": {
         "label": "print file name",
         "type": "shell",
         "command": "echo ${fileBasename}",
         "problemMatcher": []
      }
   }
   ```

7. **`${fileBasenameNoExtension}`**: The current opened file's basename without its extension.
   ```json
   {
      "settings": {
         "editor.title": "${fileBasenameNoExtension}"
      }
   }
   ```

8. **`${fileDirname}`**: The current opened file's directory.
   ```json
   {
      "settings": {
         "terminal.integrated.cwd": "${fileDirname}"
      }
   }
   ```

9. **`${fileExtname}`**: The current opened file's extension.
   ```json
   {
      "tasks": {
         "label": "print file extension",
         "type": "shell",
         "command": "echo ${fileExtname}",
         "problemMatcher": []
      }
   }
   ```

10. **`${cwd}`**: The task runner's current working directory.
    ```json
    {
       "settings": {
          "terminal.integrated.cwd": "${cwd}"
       }
    }
    ```

11. **`${lineNumber}`**: The current selected line number in the active file.
    ```json
    {
       "tasks": {
          "label": "print line number",
          "type": "shell",
          "command": "echo ${lineNumber}",
          "problemMatcher": []
       }
    }
    ```

12. **`${selectedText}`**: The currently selected text in the active file.
    ```json
    {
       "tasks": {
          "label": "print selected text",
          "type": "shell",
          "command": "echo '${selectedText}'",
          "problemMatcher": []
       }
    }
    ```

13. **`${env:Name}`**: References environment variables. Replace `Name` with the name of the environment variable.
    ```json
    {
       "settings": {
          "terminal.integrated.env.linux": {
             "PATH": "${env:PATH}:/some/other/path"
          }
       }
    }
    ```

These variables can be particularly useful in configurations like `launch.json` for debugging, `settings.json` for personalizing your VS Code environment, and `tasks.json` for defining tasks.

For the most up-to-date information and any new variables added after 2022, always refer to the official VS Code documentation.
