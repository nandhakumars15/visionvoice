## Status
- Phase 1: object detection, direction, voice alerts (done)
- Phase 2: distance estimate, priority warnings, stability filter (done)

## How it works
- YOLOv8 nano detects objects in the webcam feed
- Left / center / right zone from the box position
- Distance from the share of the frame the box covers (small objects get a boost)
- An object must appear in 2 consecutive frames before it is announced
- Close objects in the center trigger an urgent warning that jumps the speech queue

## Known limitations
- Box size is a rough distance proxy, a depth model would be more accurate
- `person` is ignored in desk testing (see `IGNORE_CLASSES`)
- Pretrained model can make false detections, so low-confidence guesses are filtered