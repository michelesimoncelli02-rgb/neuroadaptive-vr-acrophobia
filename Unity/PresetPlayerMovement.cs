/// <summary>
/// Executes the predefined movement sequence used during the offline pretest.
///
/// The script automatically guides the participant through a fixed sequence
/// of virtual height levels while sending synchronized LSL markers that
/// identify the beginning and end of each experimental trial. These markers
/// are later used to segment the recorded physiological signals during the
/// offline analysis.
///
/// The class also manages participant notifications, movement timing and
/// synchronization with the marker streaming component.
/// </summary>
using System.Collections;
using UnityEditor.VersionControl;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.XR;
using TMPro;
using UnityEngine.Timeline;

public class PresetPlayerMovement : MonoBehaviour
{
    // Height Levels: Low, Mid, High
    public Transform pointA;
    public Transform pointB;
    public Transform pointC;
    // texts on monitor and sounds
    public TMP_Text messageText;
    public AudioSource audioSource;
    public AudioClip beepSound;
    // Linking the LSL Marker hub
    public MarkerStreamer markerStreamer;

    // Duration of the height shift (adjustable if switching between the two moving modalities)
    public float movementDuration = 10f;
    // Time spent on a height level
    public float waitTime = 10f;
    // Duration of the text message
    public float messageDuration = 2f;

    // Starting point: no-fear level
    private Vector3 startPoint;

    private void Start()
    {
        startPoint = transform.position;
        StartCoroutine(MainSequence());
    }

    // Sequence of events
    private IEnumerator MainSequence()
    {
        // wait some time to let the user be familiar with the environment
        yield return new WaitForSeconds(10f);
        // marker for Experiment Start
        if (markerStreamer != null)
        {
            markerStreamer.SendMarker("Experiment_Start");
        }
        else
        {
            Debug.LogWarning("MarkerStreamer not assigned!");
        }
        // start signal
        yield return ShowMessage("START");
        // movements
        yield return MoveSequence();
        // end signal
        yield return ShowMessage("END");
        // marker for Experiment End
        if (markerStreamer != null)
        {
            markerStreamer.SendMarker("Experiment_End");
        }
        else
        {
            Debug.LogWarning("MarkerStreamer not assigned!");
        }
    }

    // Show Message on display
    private IEnumerator ShowMessage(string text)
    {
        messageText.text = text;
        messageText.gameObject.SetActive(true);

        if (audioSource && beepSound)
            audioSource.PlayOneShot(beepSound);

        yield return new WaitForSeconds(messageDuration);

        messageText.gameObject.SetActive(false);
    }

    // Smooth shifting
    private IEnumerator MoveSequence()
    {   // movements series: wait on 0 -> B -> A -> C -> B -> 0 -> A -> C -> A -> C -> B -> 0

        // forcing to stream the events of the first trial
        if (markerStreamer != null)
        {
            markerStreamer.SendMarker("Start_Trial");
        }
        else
        {
            Debug.LogWarning("MarkerStreamer not assigned!");
        }
        yield return new WaitForSeconds(waitTime);
        if (markerStreamer != null)
        {
            markerStreamer.SendMarker("End_Trial");
        }
        else
        {
            Debug.LogWarning("MarkerStreamer not assigned!");
        }
        // for all the other trials the markers´ streaming is embedded in the MoveAndWait function
        yield return MoveAndWait(pointB.position);
        yield return MoveAndWait(pointA.position);
        yield return MoveAndWait(pointC.position);
        yield return MoveAndWait(pointB.position);
        yield return MoveAndWait(startPoint);
        yield return MoveAndWait(pointA.position);
        yield return MoveAndWait(pointC.position);
        yield return MoveAndWait(pointA.position);
        yield return MoveAndWait(pointC.position);
        yield return MoveAndWait(pointB.position);
        yield return MoveAndWait(startPoint);
    }

    private IEnumerator MoveAndWait(Vector3 targetPosition)
    {   // smooth movement + waiting on a level
        yield return StartCoroutine(MoveTo(targetPosition));
        // Start Trial marker
        if (markerStreamer != null)
        {
            markerStreamer.SendMarker("Start_Trial");
        }
        else
        {
            Debug.LogWarning("MarkerStreamer not assigned!");
        }
        // Staying on the new level = Trial
        yield return new WaitForSeconds(waitTime);
        // End Trial marker
        if (markerStreamer != null)
        {
            markerStreamer.SendMarker("End_Trial");
        }
        else
        {
            Debug.LogWarning("MarkerStreamer not assigned!");
        }
    }

    private IEnumerator MoveTo(Vector3 targetPosition)
    {   // movement definition, based on starting and final positions and on movement desired duration
        Vector3 initialPosition = transform.position;
        float elapsedTime = 0f;

        while (elapsedTime < movementDuration)
        {
            transform.position = Vector3.Lerp(initialPosition, targetPosition, elapsedTime/movementDuration);
            elapsedTime += Time.deltaTime;
            yield return null;
        }

        transform.position = targetPosition;
    }
    

    // Fading + Teleporting
    /*public Image fadeImage;
     private IEnumerator MoveSequence()
    {   // movements series: wait on 0 -> A -> B -> C -> C -> B -> A -> 0
        yield return new WaitForSeconds(waitTime);
        yield return TeleportAndWait(pointA.position);
        yield return TeleportAndWait(pointB.position);
        yield return TeleportAndWait(pointC.position);
        yield return TeleportAndWait(pointC.position);
        yield return TeleportAndWait(pointB.position);
        yield return TeleportAndWait(pointA.position);
        yield return TeleportAndWait(startPoint);
    }

    private IEnumerator TeleportAndWait(Vector3 targetPosition)
    {   // Fading + teleporting + unfading
        yield return StartCoroutine(Fade(0f, 1f));

        transform.position = targetPosition;

        yield return StartCoroutine(Fade(1f, 0f));

        yield return new WaitForSeconds(waitTime);
    }

    private IEnumerator Fade(float startAlpha, float endAlpha)
    {   // progressive obscuring the image
        float elapsedTime = 0f;
        Color color = fadeImage.color;

        while (elapsedTime < movementDuration)
        {
            float alpha = Mathf.Lerp(startAlpha, endAlpha, elapsedTime/movementDuration);
            fadeImage.color = new Color(color.r, color.g, color.b, alpha);

            elapsedTime += Time.deltaTime;
            yield return null;
        }

        fadeImage.color = new Color(color.r, color.g, color.b, endAlpha);
    }
    */
}