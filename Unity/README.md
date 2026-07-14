`OnlinePlayerControl.cs` and `PresetPlayerMovement.cs` should be added to the **Player** GameObject in the Unity `MainScene`. This should be the same GameObject that has the **Main Camera** as a child. In this way, the height shifts performed by these scripts are perceived by the user as movements of the scene.

When running the **Offline Validation Experiment** or the **Calibration Pretest**, the `PresetPlayerMovement.cs` script should be enabled and the `OnlinePlayerControl.cs` script should be disabled. Conversely, when running the **Online Neuroadaptive Experiment**, `OnlinePlayerControl.cs` should be enabled and `PresetPlayerMovement.cs` should be disabled.

The **Movement Duration**, **Wait Time**, and **Message Duration** fields can be adjusted according to the experimental design. Here, they are preset to the values used in this work.

The **Message Text** field should reference a **TextMeshPro - UI** object that is a child of a **Canvas** object.

The **Audio Source** field should reference an **Audio Source** component.

The **Beep Sound** field should be assigned the preferred `.wav` audio file.

These three fields are used to notify the VR user of the beginning and the end of the experiment.
