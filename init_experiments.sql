create database experiments;
use experiments;
create table data(
id int auto_increment primary key,
temperature float,
pressure float,
catalyst_concentration float,
yeild float);

select * from data;
