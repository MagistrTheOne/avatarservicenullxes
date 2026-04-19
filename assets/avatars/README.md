# Reference portraits for ARACHNE-X-ULTRA-AVATAR (AI2V)

These images are passed to the avatar pod via `reference_image.url` in
`POST /sessions`. The pod fetches the URL, validates it as RGB, and feeds it
into ARACHNE's IdentityBank as the source portrait for AI2V (Audio-Image-to-Video)
mode.

## Available avatars

| key   | url                                                                                                  | size   |
|-------|------------------------------------------------------------------------------------------------------|--------|
| anna  | https://raw.githubusercontent.com/MagistrTheOne/avatarservicenullxes/main/assets/avatars/anna.jpg    | 97 KB  |
| denis | https://raw.githubusercontent.com/MagistrTheOne/avatarservicenullxes/main/assets/avatars/denis.png   | 406 KB |
| maxim | https://raw.githubusercontent.com/MagistrTheOne/avatarservicenullxes/main/assets/avatars/maxim.png   | 162 KB |

## Wiring

Set `AVATAR_REFERENCE_IMAGE_URL` on the gateway droplet to the chosen avatar's
`url`. The gateway forwards it verbatim to the pod for every session. To swap
identities per session, override `referenceImageUrl` in the `AvatarClient.createSession`
call (not yet exposed at the HTTP layer).

## Constraints

- Pod will reject images > 8 MB (`MAX_IMAGE_BYTES` in `inference/image_loader.py`).
- Pod fetches via plain HTTP GET — URL must be publicly reachable, no auth.
- HEAD-validated before download; ContentType must be `image/jpeg` or `image/png`.
