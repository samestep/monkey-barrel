use crate::util::{vec2, Vec2};

// counterclockwise (but in SVG it's flipped; don't be fooled!)
pub const POLYGON: [Vec2; 175] = [
    vec2(133.72229, 220.0308838),
    vec2(147.5455017, 211.2281189),
    vec2(159.0645599, 199.8363647),
    vec2(167.0275421, 189.0415802),
    vec2(177.6533203, 172.6181335),
    vec2(180.4985352, 167.282196),
    vec2(180.7580109, 164.6195221),
    vec2(180.1407471, 161.7465057),
    vec2(173.9034424, 151.2389832),
    vec2(164.8753357, 143.1319427),
    vec2(159.1849976, 140.9985962),
    vec2(144.0025024, 140.6017914),
    vec2(136.1066742, 137.0500946),
    vec2(132.0852051, 131.4690094),
    vec2(131.7050629, 125.013382),
    vec2(131.7436829, 124.7635498),
    vec2(131.8306274, 124.5314026),
    vec2(134.6401978, 109.063652),
    vec2(131.05159, 93.8299942),
    vec2(129.135788, 78.5362396),
    vec2(131.0872803, 63.3645477),
    vec2(136.4734802, 47.795845),
    vec2(144.9427795, 33.6895218),
    vec2(148.9707336, 29.4818859),
    vec2(153.7966614, 26.1994934),
    vec2(165.9481506, 19.7507362),
    vec2(178.5402832, 14.5274076),
    vec2(199.3751678, 11.9313688),
    vec2(219.3237, 18.9021339),
    vec2(233.2498627, 30.123354),
    vec2(236.7523499, 32.4397621),
    vec2(240.4578247, 33.5172806),
    vec2(250.3685303, 34.1580086),
    vec2(258.901001, 38.5632362),
    vec2(262.0629883, 43.2188301),
    vec2(263.4082031, 48.6355972),
    vec2(262.3387451, 55.1906586),
    vec2(260.3114624, 61.7556572),
    vec2(261.4460144, 68.9290619),
    vec2(265.2144165, 74.9813919),
    vec2(271.5854797, 75.1298904),
    vec2(278.1565857, 71.1438293),
    vec2(299.4916992, 43.4108505),
    vec2(319.0703735, 22.6122417),
    vec2(344.1092224, 8.8093023),
    vec2(363.2983398, 3.4813955),
    vec2(383.1354675, 2.0541916),
    vec2(400.1946716, 4.3056507),
    vec2(415.9942017, 10.6768694),
    vec2(426.8648682, 17.7609711),
    vec2(437.1668091, 25.6753445),
    vec2(450.3349915, 40.0922813),
    vec2(460.6343384, 56.8562622),
    vec2(466.7509155, 71.0173264),
    vec2(469.6730347, 85.9096146),
    vec2(469.4537048, 94.8625107),
    vec2(468.7241211, 103.8805542),
    vec2(463.8640137, 122.9504318),
    vec2(455.3978271, 140.6780853),
    vec2(451.4089966, 146.951767),
    vec2(446.3252258, 152.6337738),
    vec2(440.3565674, 154.9468079),
    vec2(435.1467896, 150.9871216),
    vec2(432.7164307, 144.7576904),
    vec2(432.5976562, 137.8714142),
    vec2(434.1716309, 124.7156906),
    vec2(434.4142761, 111.5252304),
    vec2(432.97229, 89.9981461),
    vec2(426.4047852, 69.4021072),
    vec2(418.3232422, 59.851593),
    vec2(407.7966919, 53.0267448),
    vec2(388.5993652, 48.107357),
    vec2(368.8152466, 49.115593),
    vec2(349.6053162, 54.3473816),
    vec2(333.3762817, 65.4955673),
    vec2(326.4261475, 74.210495),
    vec2(320.9422302, 84.0785828),
    vec2(306.6776123, 117.6643143),
    vec2(306.2284241, 122.3347244),
    vec2(307.5765381, 126.9374237),
    vec2(332.7004395, 171.5422363),
    vec2(343.7497864, 199.4694061),
    vec2(348.4077759, 209.4144897),
    vec2(355.6131897, 218.292923),
    vec2(368.6798401, 231.7012787),
    vec2(380.8658752, 246.1102142),
    vec2(388.4212952, 260.513916),
    vec2(387.7606812, 276.8352966),
    vec2(381.3476257, 301.6717224),
    vec2(382.2382507, 309.3421326),
    vec2(387.5112, 315.1582947),
    vec2(398.958252, 321.228302),
    vec2(409.9370117, 327.9911499),
    vec2(417.7567444, 335.882843),
    vec2(422.8139954, 345.9761963),
    vec2(422.0714111, 354.2247314),
    vec2(415.3778992, 359.2269287),
    vec2(410.7029724, 360.9933777),
    vec2(406.0625916, 362.1369019),
    vec2(399.108429, 361.5429993),
    vec2(392.2275696, 360.0759277),
    vec2(374.7958679, 354.8493347),
    vec2(335.1620483, 348.6316223),
    vec2(321.436615, 341.7218933),
    vec2(317.1513062, 326.9376221),
    vec2(318.5235291, 308.8977051),
    vec2(315.4005127, 297.6730957),
    vec2(306.1602173, 295.3896484),
    vec2(293.0257874, 296.460083),
    vec2(280.096283, 293.8604431),
    vec2(272.9387512, 294.115448),
    vec2(266.9724121, 297.8380737),
    vec2(265.8525085, 300.9356689),
    vec2(266.1832275, 304.9022827),
    vec2(268.7226562, 311.3411255),
    vec2(270.8635559, 317.7926025),
    vec2(268.5479126, 327.7814941),
    vec2(261.98349, 335.8753662),
    vec2(252.5805054, 340.7106628),
    vec2(220.599762, 344.0052795),
    vec2(204.2093658, 348.8315735),
    vec2(189.0252533, 356.9462585),
    vec2(174.5246124, 361.6273804),
    vec2(159.4064331, 358.7807617),
    vec2(155.5774384, 351.9545288),
    vec2(156.7337799, 342.8329163),
    vec2(164.6247253, 332.8400574),
    vec2(174.8295135, 325.5787964),
    vec2(184.6808319, 321.2953186),
    vec2(194.8643494, 317.5323486),
    vec2(200.770401, 314.8670349),
    vec2(205.5508118, 311.1352539),
    vec2(206.3491516, 303.8800659),
    vec2(193.4013672, 284.4219055),
    vec2(188.7812958, 269.3930054),
    vec2(190.3444824, 260.9064026),
    vec2(194.8947601, 254.1162872),
    vec2(204.0033569, 244.1230621),
    vec2(218.9357605, 233.6464996),
    vec2(222.1593781, 230.2541351),
    vec2(223.1261292, 222.7360382),
    vec2(218.8272705, 214.4584808),
    vec2(214.116806, 212.1099548),
    vec2(208.8839722, 213.7047577),
    vec2(159.9737701, 250.421524),
    vec2(141.3950958, 258.1635132),
    vec2(121.9178467, 263.3608704),
    vec2(98.3010483, 267.2456665),
    vec2(74.490242, 265.8060608),
    vec2(57.6869278, 260.6065063),
    vec2(42.8154869, 251.0285797),
    vec2(26.2494183, 237.7740936),
    vec2(22.2710304, 233.894516),
    vec2(18.8170681, 229.5418854),
    vec2(6.0074539, 206.0658264),
    vec2(2.215369, 179.5167694),
    vec2(11.7221098, 146.8741608),
    vec2(30.966774, 119.0670853),
    vec2(39.3479118, 110.971344),
    vec2(50.7653809, 106.7199402),
    vec2(57.1119232, 108.6573257),
    vec2(59.9363403, 114.516983),
    vec2(58.1782913, 124.6651306),
    vec2(53.2565956, 133.6157837),
    vec2(47.1104202, 142.3543701),
    vec2(40.4251289, 150.661087),
    vec2(32.0606995, 167.9289856),
    vec2(35.0622101, 187.1015472),
    vec2(45.2997589, 203.7035217),
    vec2(59.1971092, 217.0593109),
    vec2(73.1209183, 226.2915802),
    vec2(89.1017532, 230.886261),
    vec2(100.6980515, 231.1419678),
    vec2(112.1001358, 228.3657379),
    vec2(133.72229, 220.0308838), // duplicated from the first point
];
