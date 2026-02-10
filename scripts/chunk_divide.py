import json

# text1 = {"osm_type": "node", "osm_id": 1574399948, "name": "Kawiarnia Palmiarnia", "lat": 51.7602918, "lon": 19.4796323, "text_chunk": "Kawiarnia Palmiarnia — cafe Adres: Aleja Marszałka Józefa Piłsudskiego 61, 90-368, Łódź. Godziny otwarcia: Tu-Fr: 12:00-18:00; Sa-Su: 10:00-18:00. Telefon: +48 79 335 3588. Strona: https://www.facebook.com/kawiarniapalmiarnia/. Udogodnienia: Miejsca na zewnątrz: yes; Na wynos: no; Dostawa: no. Współrzędne: 51.760292, 19.479632. OSM: https://www.openstreetmap.org/node/1574399948. Dodatkowe tagi: discount:citycard: yes."}
# text = {"osm_type": "node", "osm_id": 1820217350, "name": "Restauracja Zielona", "lat": 51.7806506, "lon": 19.4479858, "text_chunk": "Restauracja Zielona — restaurant Kuchnia: pasta, burger, asian, pasta;burger;asian. Adres: Drewnowska 58, 91-002, Łódź. Godziny otwarcia: Mo-Sa  10:00-22:00; Su 10:00-20:00. Telefon: +48 42 632 16 96. Strona: http://www.greenway.pl. Udogodnienia: Dostęp dla wózków: limited; Miejsca wewnątrz: yes; Miejsca na zewnątrz: yes; Internet: wlan. Diety i alergeny: wegetariańskie: only; wegańskie: yes; bezglutenowe: yes. Współrzędne: 51.780651, 19.447986. OSM: https://www.openstreetmap.org/node/1820217350. Dodatkowe tagi: contact:fax: +48 42 632 16 96; level: 0."}

ready_for_embd = []
output_file = "output_files/lodz_restaurants_cafes_ready_for_embd.jsonl"

with open("output_files/lodz_restaurants_cafes_chunks.jsonl", "r", encoding="utf-8") as file:
    for line in file:
        record = json.loads(line)
        chunk_text = record["text_chunk"]
        # wycinam nazwe miejsca by zabezpieczyć rozdzielenie na złe chunki gdy w nazwie też jest '.'
        chunk_text_without_name = chunk_text.split(" — ", 1)[1]
        res = chunk_text_without_name.split(". ")
        # print("RESSS", res)
        osm_id = record["osm_id"]
        name = record["name"]

        chunks = [x for x in res if not x.strip().startswith("OSM:")]
        # print(chunks)
        # print("TUTUATJATJ!!", chunks[0])
        place_type = chunks[0].split(" ")[0].strip()
        chunks[0] = chunks[0].split(place_type)[1].strip()
        
        # print("PO WSZYSTKIC HROZNYCH ASGAS", chunks)
            
        # print("PO", chunks)

        # print(chunks)

        record = {}

        record["oms_id"] = osm_id
        record["name"] = name
        record["type"] = place_type

        for content in chunks:
            if ":" in content:
                key, value = content.split(":", 1)
                key = key.strip()
                value = value.strip()
                record[key] = value
            else:
                record["description"] = content.strip()
            
        print("\nRECORD!!!", record)        

        ready_for_embd.append(record)

# print("READYYYYYYY", ready_for_embd)

with open(output_file, "w", encoding="utf-8") as f:
    for record in ready_for_embd:
        json.dump(record, f, ensure_ascii=False)
        f.write("\n")