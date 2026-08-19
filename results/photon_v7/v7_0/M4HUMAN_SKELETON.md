# M4Human SMPL-X 22-Joint Body Skeleton Topology & Coordinate System

## Skeleton Topology (22 Body Joints)

| Joint ID | Joint Name | Parent ID | Parent Name | Coordinate System | Visibility |
| :---: | :--- | :---: | :--- | :--- | :---: |
| 0 | Pelvis (Root) | -1 | None (Root Center) | Radar Frame (meters) | Always Visible |
| 1 | Left Hip | 0 | Pelvis | Radar Frame (meters) | Always Visible |
| 2 | Right Hip | 0 | Pelvis | Radar Frame (meters) | Always Visible |
| 3 | Spine 1 | 0 | Pelvis | Radar Frame (meters) | Always Visible |
| 4 | Left Knee | 1 | Left Hip | Radar Frame (meters) | Always Visible |
| 5 | Right Knee | 2 | Right Hip | Radar Frame (meters) | Always Visible |
| 6 | Spine 2 | 3 | Spine 1 | Radar Frame (meters) | Always Visible |
| 7 | Left Ankle | 4 | Left Knee | Radar Frame (meters) | Always Visible |
| 8 | Right Ankle | 5 | Right Knee | Radar Frame (meters) | Always Visible |
| 9 | Spine 3 | 6 | Spine 2 | Radar Frame (meters) | Always Visible |
| 10 | Left Foot | 7 | Left Ankle | Radar Frame (meters) | Always Visible |
| 11 | Right Foot | 8 | Right Ankle | Radar Frame (meters) | Always Visible |
| 12 | Neck | 9 | Spine 3 | Radar Frame (meters) | Always Visible |
| 13 | Left Collar | 9 | Spine 3 | Radar Frame (meters) | Always Visible |
| 14 | Right Collar | 9 | Spine 3 | Radar Frame (meters) | Always Visible |
| 15 | Head | 12 | Neck | Radar Frame (meters) | Always Visible |
| 16 | Left Shoulder | 13 | Left Collar | Radar Frame (meters) | Always Visible |
| 17 | Right Shoulder | 14 | Right Collar | Radar Frame (meters) | Always Visible |
| 18 | Left Elbow | 16 | Left Shoulder | Radar Frame (meters) | Always Visible |
| 19 | Right Elbow | 17 | Right Shoulder | Radar Frame (meters) | Always Visible |
| 20 | Left Wrist | 18 | Left Elbow | Radar Frame (meters) | Always Visible |
| 21 | Right Wrist | 19 | Right Elbow | Radar Frame (meters) | Always Visible |

## Kinematic Chain & Bone Connectivity
- Spine Chain: 0 -> 3 -> 6 -> 9 -> 12 -> 15
- Left Arm: 9 -> 13 -> 16 -> 18 -> 20
- Right Arm: 9 -> 14 -> 17 -> 19 -> 21
- Left Leg: 0 -> 1 -> 4 -> 7 -> 10
- Right Leg: 0 -> 2 -> 5 -> 8 -> 11
