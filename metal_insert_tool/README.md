# Read from DICOM(dcm) file, save numpy, img files

## 1. Run

>pip install -r `requirements.txt` <br>
run `main.py`

## 2. Usage
MA Path : <br>
Non MA Path : <br>
X Y : <br>
Zoom : Zoom Metal in by the value, input real number <br>
Angle(0~360) : Rotate metal. input degree<br>
Set Image Button : Load dicom file and shows metal inserted image in by the parameters <br>
Save Image Button : Save numpy, and image file in `./inserted/` directory. File name is "Non MA Pateient Number_MA Number_File Count"<br>

## 3. Make execute file

>> not running
Need Windows 10 SDK([Download Page]((https://developer.microsoft.com/en-us/windows/downloads/windows-10-sdk)<br>
[VC++ Redistri](https://www.microsoft.com/en-US/download/details.aspx?id=48145)<br>

> pyinstaller -F -w -p "C:\Program Files\Python36\Lib\site-packages\PyQt5\Qt\bin;C:\Program Files (x86)\Windows Kits\10\Redist\ucrt\DLLs\x64" main.py


