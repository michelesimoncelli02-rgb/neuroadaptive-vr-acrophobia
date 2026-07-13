using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class HeightVolume : MonoBehaviour
{
    public Transform player;
    public AudioSource audioSrc;

    public float minHeight = 0f;
    public float maxHeight = 200f;

    public float minVolume = 0f;
    public float maxVolume = 1f;

    // Update is called once per frame
    void Update()
    {
        // get the player current y position
        float h = player.position.y;

        // clamp the y value between minHeight and maxHeight
        float t = Mathf.Clamp(h, minHeight, maxHeight);

        // volume calculation
        audioSrc.volume = Mathf.InverseLerp(minHeight, maxHeight, t) * (maxVolume - minVolume) + minVolume;
    }
}
