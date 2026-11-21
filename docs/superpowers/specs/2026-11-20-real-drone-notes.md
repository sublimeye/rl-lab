# A real drone: early notes (no purchase, no build)

Asked for on 2026-11-20 as early exploration: what would it take to put a trained pilot, sensors
and behaviour on real hardware. Prices and availability are rough and change; check before buying.

## What already carries over from posts 10–11

- **Stick-level actions in angle mode.** Most small drones accept "virtual stick" commands:
  roll/pitch/yaw/throttle, each -100..100. That's our action space plus two more axes.
- **Sensor noise, a 50 ms delay, wind, payload, a sagging battery.** These are the usual reasons
  a simulated pilot fails on hardware, and the post-11 pilots already train with all of them.
- **A coarse depth "camera".** Distance sensors in a few directions (time-of-flight) are close to
  our 32 rays, just fewer of them.
- **The networks are tiny.** Post 10: 5,250 numbers (21 KB as float32). Post 11's
  exact-position pilot is about 27,000 numbers, the camera pilot about 50,000 (under 200 KB as
  float32). Both fit on a drone's own microcontroller; one decision is tens of thousands of
  multiply-adds, well under a millisecond on a Cortex-M4-class chip.

## What doesn't carry over

- **2D to 3D.** The simulator needs roll and pitch (two tilts), yaw and a third position axis.
  The physics are the same kind; the network gets more inputs and four outputs.
- **Knowing where it is.** The simulator gives positions for free (or a noisy "GPS" fix). Indoors,
  a real drone needs optical flow plus a height sensor, a positioning system, or markers seen by
  a camera.
- **Recognising the pad.** On hardware: a printed marker (e.g. an AprilTag) and a camera.
- **Birds.** Indoors, stand-ins: a person walking a foam ball on a pole, or a second drone.

## Candidate platforms

| Platform | Roughly | Where the network runs | Notes |
| --- | --- | --- | --- |
| Bitcraze Crazyflie 2.1 (+ Flow deck, Multi-ranger deck) | €200–500 with decks | On the drone (STM32F405) or a laptop over radio | Open firmware, made for research; published work has run simulation-trained RL pilots on its own chip. Multi-ranger: 5 distance sensors (front, back, left, right, up); Flow deck: optical flow + height. 27 g, safe indoors. |
| Ryze/DJI Tello (EDU) | ~$100–200 if still available | Laptop, over Wi-Fi (~100 ms round trip) | Its SDK has an `rc` command that is literally four virtual sticks. Closed firmware; downward vision for hovering; video to the laptop. The quickest way to see a sim pilot fly. |
| Espressif ESP-Drone (ESP32) | ~$30–60 in parts | On the drone (ESP32) or a phone/laptop over Wi-Fi | DIY and open; cheapest, more tinkering. |
| A small wheeled robot with distance sensors | ~$50–150 | On board (e.g. ESP32/RP2040) or a laptop | Not a drone, but the safest first step for "model + sensors + behaviour" in the real world. |

## A possible path

1. **3D simulator** (vectorised, as in post 11): roll, pitch, yaw, three axes, the same noise,
   delay and domain randomisation, plus the target platform's real command rate and latency.
2. **Tello first**: the policy on the laptop sends `rc` commands at 20 Hz; hovering and a hop
   between two marked boxes. Measure the real latency and add it to the simulator.
3. **Crazyflie second**: the same pilot on the drone's own microcontroller (no Wi-Fi delay), with
   the Flow and Multi-ranger decks standing in for the depth camera.
4. Safety: indoors, prop guards or a net, small sub-250 g drones; check local rules before flying
   outdoors.
