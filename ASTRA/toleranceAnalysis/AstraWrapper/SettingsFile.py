
class SettingsFile:

    def loadInputData(self, lines):
        library = {"": ""}

        for line in lines:
            if "=" in line:
                key = str(line).split("=")[0].replace("!", "").strip()
                val = str(line).split("=")[1].strip()
                library[key] = val

    def __init__(self, nameOfFile):
        try:
            lines = []
            if ".in" in nameOfFile:
                with open(nameOfFile,'r') as file:
                    lines = file.readlines()
            else:
                with open(nameOfFile + ".in", 'r') as file:
                    lines = file.readlines()

            self.fileName = nameOfFile
            self.library = {}  # Assuming you forgot to declare this earlier
            self.loadInputData(lines)

        except FileNotFoundError:
            raise FileNotFoundError("The file " + nameOfFile + ".in was not found or there was a problem with loading data.")

    def changeInputData(self, tag, newVar):
        try:
            self.library[tag] = newVar

            # Open the file for reading
            lines = []
            with open(self.fileName + ".in", 'r') as file:
                lines = file.readlines()

            # Prepare the replacement string
            replacement = " " + tag + "=" + str(newVar) + "\n"

            # Iterate over the lines to find and replace the target line
            for i, line in enumerate(lines):
                if tag in line and not "&" in line:
                    lines[i] = replacement
                    break  # Assuming there's only one occurrence to replace

            # Write the modified lines back to the file
            with open(self.fileName + ".in", 'w') as file:
                file.writelines(lines)

            return True

        except FileNotFoundError:
            raise FileNotFoundError("The file " + self.fileName + ".in was not found.")

        except Exception as e:
            raise ValueError(f"An error occurred when trying to change '{tag}' to variable '{newVar}': {e}")

    def enable(self, tag):
        try:
            # Open the file for reading
            with open(self.fileName + ".in", 'r') as file:
                lines = file.readlines()

            # Iterate over the lines to find the target line
            for i, line in enumerate(lines):
                if tag in line:
                    if "!" in line:
                        lines[i] = line.replace("!", "")
                        break

            # Write the modified lines back to the file
            with open(self.fileName + ".in", 'w') as file:
                file.writelines(lines)

            return True

        except FileNotFoundError:
            raise FileNotFoundError("The file " + self.fileName + ".in was not found.")

        except Exception as e:
            raise ValueError(f"An error occurred when trying to enable '{tag}': {e}")

    def disable(self, tag):
        try:
            # Open the file for reading
            with open(self.fileName + ".in", 'r') as file:
                lines = file.readlines()

            for i, line in enumerate(lines):
                if tag in line:
                    if "!" in line:
                        break
                    else:
                        lines[i] = "!" + lines[i]
                        break

            with open(self.fileName + ".in", 'w') as file:
                file.writelines(lines)

            return True

        except FileNotFoundError:
            raise FileNotFoundError("The file " + self.fileName + ".in was not found.")

        except Exception as e:
            raise ValueError(f"An error occurred when trying to change '{tag}' to variable '{newVar}': {e}")

    def readOption(self, tag):
        try:
            # Open the file for reading
            with open(self.fileName + ".in", 'r') as file:
                lines = file.readlines()

            # Iterate over the lines to find and replace the target line
            setting = ''
            for i, line in enumerate(lines):
                if tag in line:
                    setting = line.split("=")
                    return setting[-1]

            raise ValueError(f"No occurence of tag {tag} in {self.fileName}.in.")

        except FileNotFoundError:
            raise FileNotFoundError("The file " + self.fileName + ".in was not found.")

        except Exception as e:
            raise ValueError(f"An error occurred when trying to read option '{tag}': {e}")

    def checkOption(self, tag):
        # check if this option is enabled = True or disabled = False
        with open(self.fileName + ".in", 'r') as file:
            lines = file.readlines()

        for i, line in enumerate(lines):
            if tag in line:
                if "!" in line:
                    return False
                else:
                    return True
	    