; Inno Setup script for ZEUS. Build the app first, then compile this:
;
;   pyinstaller zeus.spec
;   iscc installer\zeus.iss
;
; Output lands in installer\Output\ZEUS-<version>-setup.exe.
; Needs Inno Setup 6.3 or newer (x64compatible, and the scaled wizard images).

#define AppName "ZEUS"
#define AppPublisher "Aizad"
#define AppExeName "ZEUS.exe"
; Read back off the built exe, which zeus.spec stamps from pyproject.toml, so
; the version is written down in exactly one place. Compiling before building
; the app fails here, which is the intended order.
#define AppVersion GetStringFileInfo("..\dist\ZEUS\ZEUS.exe", "ProductVersion")

[Setup]
; Identifies the app across upgrades and uninstalls -- must never change once
; a build has shipped, or upgrades install alongside instead of over.
AppId={{0582B8B4-604C-42D4-B945-65D3051E6704}
AppName={#AppName}
AppVersion={#AppVersion}
AppPublisher={#AppPublisher}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
UninstallDisplayName={#AppName}
UninstallDisplayIcon={app}\{#AppExeName}
OutputDir=Output
OutputBaseFilename={#AppName}-{#AppVersion}-setup
SetupIconFile=..\gui\assets\icon.ico

; The payload is ~370 MB of mostly-compressible DLLs, so solid LZMA2 is worth
; the slower compile: it takes the installer to roughly a third of that.
Compression=lzma2/ultra64
SolidCompression=yes
LZMANumBlockThreads=4
; ISCC.exe is 32-bit and runs out of address space compressing a payload this
; large at ultra64. This hands the work to islzma64.exe instead.
LZMAUseSeparateProcess=yes

; Falls back to a per-user install under {localappdata}\Programs when the user
; has no admin rights, rather than refusing to run.
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog

; PySide6 6.11 needs Windows 10 1809 or newer.
MinVersion=10.0.17763
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible

; Shown as the acceptance page. GPLv3 does not require accepting terms to run
; the program, but displaying it is conventional -- drop this line to skip the
; page while still installing the file below.
LicenseFile=..\LICENSE

WizardStyle=modern
WizardImageFile=wizard-large-164.bmp,wizard-large-246.bmp,wizard-large-328.bmp
WizardSmallImageFile=wizard-small-55.bmp,wizard-small-110.bmp,wizard-small-164.bmp
DisableProgramGroupPage=yes
; Offers to close a running copy on upgrade instead of failing on locked files.
CloseApplications=yes
RestartApplications=no

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
; The whole PyInstaller onedir tree: ZEUS.exe plus its _internal folder.
Source: "..\dist\ZEUS\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs
; Also at the install root: PyInstaller buries its copies under _internal\,
; where nobody looking for the licence terms would think to check.
Source: "..\LICENSE"; DestDir: "{app}"; Flags: ignoreversion
Source: "..\THIRD-PARTY-NOTICES.md"; DestDir: "{app}"; Flags: ignoreversion

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent

; Deliberately no [UninstallDelete] for the crash log or the saved theme under
; {userappdata}\EIS Batch Analysis: those are the user's, and a reinstall
; should find their settings intact.
