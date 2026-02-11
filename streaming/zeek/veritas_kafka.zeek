##! Veritas — Zeek-kafka plugin configuration.
##!
##! Routes Zeek connection, SSL, X.509, HTTP, and DNS logs to their
##! respective Kafka topics for real-time classification by the Veritas
##! Spark Structured Streaming pipeline.
##!
##! Prerequisites:
##!   - zeek-kafka plugin installed (https://github.com/SeisoLLC/zeek-kafka)
##!   - Kafka broker(s) reachable from the Zeek sensor
##!
##! Usage:
##!   zeek -i <interface> veritas_kafka.zeek

@load packages/zeek-kafka

redef Kafka::kafka_conf = table(
    ["metadata.broker.list"] = "localhost:9092",
    ["compression.codec"]    = "lz4",
    ["batch.num.messages"]   = "500",
    ["queue.buffering.max.ms"] = "100"
);

redef Kafka::logs_to_send = set(
    Conn::LOG,
    SSL::LOG,
    X509::LOG,
    HTTP::LOG,
    DNS::LOG
);

redef Kafka::topic_name = "";

redef Kafka::tag_json = F;

# Route each log type to its own topic.
event zeek_init()
{
    Kafka::set_topic(Conn::LOG,  "veritas.conn");
    Kafka::set_topic(SSL::LOG,   "veritas.ssl");
    Kafka::set_topic(X509::LOG,  "veritas.x509");
    Kafka::set_topic(HTTP::LOG,  "veritas.http");
    Kafka::set_topic(DNS::LOG,   "veritas.dns");
}
