; Inno Setup script for Datamarkin Windows installer.
;
; Prerequisites:
;   1. Build with: python build.py
;   2. Install Inno Setup: https://jrsoftware.org/isinfo.php
;   3. Compile this script: iscc installer.iss
;
; Output: Output/Datamarkin-{version}-Setup.exe

#define MyAppName "Datamarkin"
#define MyAppVersion "0.1.0"
#define MyAppPublisher "Datamarkin"
#define MyAppURL "https://datamarkin.com"
#define MyAppExeName "Datamarkin.exe"

[Setup]
AppId={{A3F8D2E1-5B6C-4A7D-9E0F-1C2D3E4F5A6B}
AppName={#MyAppName}
AppVersion={#MyAppVersion}
AppPublisher={#MyAppPublisher}
AppPublisherURL={#MyAppURL}
DefaultDirName={autopf}\{#MyAppName}
DefaultGroupName={#MyAppName}
DisableProgramGroupPage=yes
OutputDir=Output
OutputBaseFilename=Datamarkin-{#MyAppVersion}-Setup
Compression=lzma2
SolidCompression=yes
WizardStyle=modern
PrivilegesRequired=lowest
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\Datamarkin\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"
Name: "{autodesktop}\{#MyAppName}"; Filename: "{app}\{#MyAppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#MyAppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(MyAppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
